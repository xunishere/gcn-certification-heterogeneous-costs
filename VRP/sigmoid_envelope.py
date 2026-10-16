"""Sound affine envelopes for sigmoid on arbitrary finite intervals.

The only soundness domain in this module is the supplied *outer* interval.
Under-domain samples may propose slopes and may be used as a line-selection
objective, but every intercept is recalibrated analytically on the full outer
interval before a line is returned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

ENVELOPE_VERSION = "sigmoid-stationary-intercept-v1"


def sigmoid(value: float) -> float:
    """Overflow-safe scalar logistic sigmoid."""
    value = float(value)
    if not math.isfinite(value):
        if value == math.inf:
            return 1.0
        if value == -math.inf:
            return 0.0
        raise ValueError("sigmoid input must not be NaN")
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def sigmoid_derivative(value: float) -> float:
    """Numerically stable ``sigmoid(x) * (1-sigmoid(x))``."""
    value = float(value)
    if not math.isfinite(value):
        if abs(value) == math.inf:
            return 0.0
        raise ValueError("sigmoid derivative input must not be NaN")
    exponential = math.exp(-value) if value >= 0.0 else math.exp(value)
    return exponential / ((1.0 + exponential) ** 2)


def _stationary_points_for_slope(slope: float) -> tuple[float, ...]:
    """Return all finite roots of ``sigmoid'(x) == slope``.

    For ``0 < a < 1/4`` the two roots are obtained through the logistic
    probability roots of ``p(1-p)=a``.  The low root uses a rationalized
    expression to avoid cancellation when ``a`` is small.
    """
    slope = float(slope)
    if not math.isfinite(slope):
        raise ValueError("slope must be finite")
    if slope <= 0.0 or slope > 0.25:
        return ()
    if slope == 0.25:
        return (0.0,)
    discriminant = math.sqrt(max(0.0, 1.0 - 4.0 * slope))
    log_probability_low = (math.log(2.0) + math.log(slope)
                           - math.log1p(discriminant))
    probability_low = math.exp(log_probability_low)
    low = log_probability_low - math.log1p(-probability_low)
    # sigmoid' is even, so using -low avoids loss of 1-probability_low when
    # the low logistic root is below binary64 resolution around one.
    return (low, -low)


@dataclass(frozen=True)
class CalibratedIntercepts:
    slope: float
    lower_intercept: float
    upper_intercept: float
    raw_lower_intercept: float
    raw_upper_intercept: float
    critical_points: tuple[float, ...]

    @property
    def maximum_vertical_width(self) -> float:
        return self.upper_intercept - self.lower_intercept


def calibrate_intercepts(lower: float, upper: float, slope: float) -> CalibratedIntercepts:
    """Compute tight affine intercepts for one slope on ``[lower, upper]``.

    Let ``g(x)=sigmoid(x)-a*x``.  Its extrema on a compact interval occur at
    the two endpoints or at roots of ``sigmoid'(x)=a``.  Evaluating exactly
    this finite set yields the mathematically tight intercepts for the chosen
    slope. Returned safety intercepts include a conservative binary64
    evaluation-error allowance and a final outward ULP; the unpadded tight
    values are retained in ``raw_*``.
    """
    lower, upper, slope = float(lower), float(upper), float(slope)
    if not all(math.isfinite(value) for value in (lower, upper, slope)):
        raise ValueError("interval endpoints and slope must be finite")
    if lower > upper:
        raise ValueError("lower endpoint exceeds upper endpoint")
    points = [lower, upper]
    for point in _stationary_points_for_slope(slope):
        if lower <= point <= upper:
            points.append(point)
    points = sorted(set(points))
    values = [sigmoid(point) - slope * point for point in points]
    raw_lower, raw_upper = min(values), max(values)
    # Account for exp/divide/product/subtraction roundoff before the final ULP
    # step. This matters when sigmoid(x) and a*x nearly cancel and the
    # intercept itself is much smaller than either operand.
    epsilon = 16.0 * math.ulp(1.0)
    errors = [epsilon * (1.0 + abs(slope * point)) for point in points]
    safe_lower = math.nextafter(
        min(value - error for value, error in zip(values, errors)), -math.inf
    )
    safe_upper = math.nextafter(
        max(value + error for value, error in zip(values, errors)), math.inf
    )
    return CalibratedIntercepts(
        slope=slope,
        lower_intercept=safe_lower,
        upper_intercept=safe_upper,
        raw_lower_intercept=raw_lower,
        raw_upper_intercept=raw_upper,
        critical_points=tuple(points),
    )


@dataclass(frozen=True)
class CandidateSlope:
    slope: float
    source: str


@dataclass(frozen=True)
class AffineLine:
    slope: float
    intercept: float
    source: str
    side: str
    selection_mean_gap: float
    outer_max_gap: float
    outer_lower: float
    outer_upper: float

    def value(self, x: float) -> float:
        return self.slope * float(x) + self.intercept

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnvelopeSelection:
    lower: AffineLine
    upper: AffineLine
    method: str
    used_under_domain: bool
    fallback_reason: str | None
    outer_lower: float
    outer_upper: float
    under_lower: float | None
    under_upper: float | None
    candidate_count: int
    selection_point_count: int
    version: str = ENVELOPE_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def _secant_slope(lower: float, upper: float) -> float:
    if lower == upper:
        return sigmoid_derivative(lower)
    return (sigmoid(upper) - sigmoid(lower)) / (upper - lower)


def _append_unique(candidates: list[CandidateSlope], slope: float, source: str) -> None:
    if not math.isfinite(float(slope)):
        return
    slope = float(slope)
    for candidate in candidates:
        if abs(candidate.slope - slope) <= 1e-15 * max(1.0, abs(slope)):
            return
    candidates.append(CandidateSlope(slope, source))


def outer_candidate_slopes(lower: float, upper: float) -> list[CandidateSlope]:
    """Slopes available to the outer-only baseline."""
    lower, upper = float(lower), float(upper)
    if lower > upper:
        raise ValueError("lower endpoint exceeds upper endpoint")
    middle = lower + 0.5 * (upper - lower)
    candidates: list[CandidateSlope] = []
    _append_unique(candidates, 0.0, "outer/constant")
    _append_unique(candidates, _secant_slope(lower, upper), "outer/secant")
    _append_unique(candidates, sigmoid_derivative(lower), "outer/tangent-lower")
    _append_unique(candidates, sigmoid_derivative(upper), "outer/tangent-upper")
    _append_unique(candidates, sigmoid_derivative(middle), "outer/tangent-midpoint")
    if lower <= 0.0 <= upper:
        _append_unique(candidates, 0.25, "outer/tangent-zero")
    return candidates


def under_candidate_slopes(
    under_lower: float,
    under_upper: float,
    under_values: Sequence[float],
) -> list[CandidateSlope]:
    """Additional slopes proposed by feasible Monte Carlo preactivations."""
    under_lower, under_upper = float(under_lower), float(under_upper)
    middle = under_lower + 0.5 * (under_upper - under_lower)
    candidates: list[CandidateSlope] = []
    _append_unique(candidates, _secant_slope(under_lower, under_upper),
                   "under/secant-endpoints")
    _append_unique(candidates, sigmoid_derivative(under_lower),
                   "under/tangent-lower")
    _append_unique(candidates, sigmoid_derivative(under_upper),
                   "under/tangent-upper")
    _append_unique(candidates, sigmoid_derivative(middle),
                   "under/tangent-midpoint")
    finite_values = sorted(float(value) for value in under_values
                           if math.isfinite(float(value)))
    if finite_values:
        for label, index in (("q25", (len(finite_values) - 1) // 4),
                             ("median", (len(finite_values) - 1) // 2),
                             ("q75", 3 * (len(finite_values) - 1) // 4)):
            _append_unique(candidates, sigmoid_derivative(finite_values[index]),
                           f"under/tangent-{label}")
    return candidates


def _selection_gap(line_slope: float, intercept: float, side: str,
                   points: Sequence[float]) -> float:
    gaps = []
    for point in points:
        exact = sigmoid(point)
        affine = line_slope * point + intercept
        gap = exact - affine if side == "lower" else affine - exact
        # One-ULP outward calibration can still accumulate a tiny signed error
        # in a second arithmetic expression; retain a non-negative diagnostic.
        gaps.append(max(0.0, gap))
    return math.fsum(gaps) / len(gaps)


def _choose_line(
    candidates: Sequence[CandidateSlope],
    lower: float,
    upper: float,
    side: str,
    selection_points: Sequence[float],
) -> AffineLine:
    best: tuple[float, int, AffineLine] | None = None
    for order, candidate in enumerate(candidates):
        calibrated = calibrate_intercepts(lower, upper, candidate.slope)
        intercept = (calibrated.lower_intercept if side == "lower"
                     else calibrated.upper_intercept)
        mean_gap = _selection_gap(candidate.slope, intercept, side, selection_points)
        line = AffineLine(
            slope=candidate.slope,
            intercept=intercept,
            source=candidate.source,
            side=side,
            selection_mean_gap=mean_gap,
            outer_max_gap=calibrated.maximum_vertical_width,
            outer_lower=lower,
            outer_upper=upper,
        )
        key = (mean_gap, order, line)
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None:
        raise RuntimeError("no affine candidate was generated")
    return best[2]


def select_envelope(
    outer_lower: float,
    outer_upper: float,
    *,
    method: str = "baseline",
    under_values: Iterable[float] | None = None,
    variation_tolerance: float = 1e-12,
) -> EnvelopeSelection:
    """Select one safe lower and upper line for a sigmoid neuron.

    ``baseline`` generates candidates and selection anchors only from the
    outer interval.  ``dual`` adds slopes generated from feasible under-domain
    samples and minimizes mean line gap at those samples.  Candidate inclusion
    does not grant validity: every candidate is recalibrated on the complete
    outer interval by :func:`calibrate_intercepts`.
    """
    if method not in ("baseline", "dual"):
        raise ValueError("method must be 'baseline' or 'dual'")
    lower, upper = float(outer_lower), float(outer_upper)
    if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
        raise ValueError("outer interval must be finite and ordered")
    candidates = outer_candidate_slopes(lower, upper)
    outer_points = [lower, lower + 0.5 * (upper - lower), upper]
    if lower < 0.0 < upper:
        outer_points.append(0.0)
    selection_points = outer_points
    used_under = False
    fallback_reason = None
    under_lower = under_upper = None

    if method == "dual":
        values = ([] if under_values is None else
                  [float(value) for value in under_values
                   if math.isfinite(float(value))])
        # Samples are numerical evaluations of feasible attacks.  Clipping an
        # O(ulp) propagation discrepancy keeps diagnostics inside the proven
        # outer interval; a substantive violation is rejected.
        if values:
            tolerance = 1e-9 * max(1.0, abs(lower), abs(upper))
            if min(values) < lower - tolerance or max(values) > upper + tolerance:
                raise ValueError("under-domain sample lies outside outer interval")
            values = [min(upper, max(lower, value)) for value in values]
            under_lower, under_upper = min(values), max(values)
        if not values:
            fallback_reason = "no_under_samples"
        elif under_upper - under_lower <= variation_tolerance * max(
                1.0, abs(under_lower), abs(under_upper)):
            fallback_reason = "under_domain_no_variation"
        else:
            for candidate in under_candidate_slopes(under_lower, under_upper, values):
                _append_unique(candidates, candidate.slope, candidate.source)
            selection_points = values
            used_under = True

    lower_line = _choose_line(
        candidates, lower, upper, "lower", selection_points
    )
    upper_line = _choose_line(
        candidates, lower, upper, "upper", selection_points
    )
    return EnvelopeSelection(
        lower=lower_line,
        upper=upper_line,
        method=method,
        used_under_domain=used_under,
        fallback_reason=fallback_reason,
        outer_lower=lower,
        outer_upper=upper,
        under_lower=under_lower,
        under_upper=under_upper,
        candidate_count=len(candidates),
        selection_point_count=len(selection_points),
    )
