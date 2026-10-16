"""Controlled exhaustive real-model checks for a completed deep experiment.

This enumerates a restricted 12-variable attack subset. It is a soundness
check, never a claim to solve the complete formal threat exactly.
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset_registry import dataset_spec, load_dataset
from .deep_gcn import TrainingHyperparameters, train_or_load_deep
from .deep_verifier import build_computation_graph
from .real_experiments import masks


def run(args):
    formal_root=Path(args.formal_root); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    formal=pd.read_csv(formal_root/"nodes.csv"); records=[]
    for dataset in ("Cora","CiteSeer"):
        data=load_dataset(dataset,args.cache_dir)[0]
        for depth in (2,3):
            manifest=json.loads((formal_root/"shards"/f"{dataset}-s0-d{depth}"/
                "targets"/dataset/"seed-0"/f"depth-{depth}.json").read_text(encoding="utf-8"))
            targets=manifest["targets"]; chosen=[targets[0],targets[len(targets)//2],targets[-1]]
            model_run=train_or_load_deep(
                data,dataset,0,"cpu",depth=depth,hidden_dims=[16]*(depth-1),bias=True,
                hyperparameters=TrainingHyperparameters(500,.01,5e-4,100),masks=masks(data,0),
                model_cache_dir=args.model_cache_dir)
            weights,biases=model_run.model.numpy_parameters()
            clean_x=model_run.x.detach().cpu().double().numpy()
            specification=dataset_spec(dataset)
            for target in chosen:
                graph=build_computation_graph(model_run.adjacency,target,depth)
                source_nodes=np.asarray(graph.candidate_sources,dtype=int); local=clean_x[source_nodes]
                clean_first=graph.adjacency_blocks[0]@local@weights[0]+biases[0]
                score=np.max(graph.adjacency_blocks[0],axis=0)[:,None]*np.max(np.abs(weights[0]),axis=1)[None,:]
                flat=score.ravel(); count=min(12,flat.size); top=np.argpartition(flat,-count)[-count:]
                top=top[np.argsort(flat[top])[::-1]]
                variables=[(int(index//score.shape[1]),int(index%score.shape[1])) for index in top]
                def logits_for(flips):
                    first=clean_first.copy()
                    for source,feature in flips:
                        first += (graph.adjacency_blocks[0][:,source,None]
                                  *(1-2*local[source,feature])*weights[0][feature][None,:])
                    previous=1/(1+np.exp(-np.clip(first,-745,709)))
                    for layer in range(2,depth+1):
                        previous=graph.adjacency_blocks[layer-1]@previous@weights[layer-1]+biases[layer-1]
                        if layer<depth: previous=1/(1+np.exp(-np.clip(previous,-745,709)))
                    return previous[0]
                clean=logits_for(()); prediction=int(np.argmax(clean)); rivals=[i for i in range(len(clean)) if i!=prediction]
                clean_margin=min(float(clean[prediction]-clean[r]) for r in rivals)
                worst=math.inf; worst_flips=(); attacks=0
                for size in range(4):
                    for subset in itertools.combinations(variables,size):
                        logits=logits_for(subset); margin=min(float(logits[prediction]-logits[r]) for r in rivals); attacks+=1
                        if margin<worst: worst,worst_flips=margin,subset
                selected=formal[(formal.dataset==dataset)&(formal.seed==0)&(formal.depth==depth)&
                                (formal.target==target)&(formal["mode"]=="legacy")&
                                (formal.q==specification.main_q)&
                                (formal.Q0==specification.main_q0)]
                for method in ("baseline","dual"):
                    row=selected[selected.method==method].iloc[0]; lower=float(row.lp_lower_bound)
                    records.append({
                        "dataset":dataset,"seed":0,"depth":depth,"target":target,"method":method,
                        "q":specification.main_q,"Q":specification.main_q0,
                        "controlled_variables":count,
                        "controlled_attacks_evaluated":attacks,"clean_margin":clean_margin,
                        "controlled_exact_worst_margin":worst,"lp_lower_bound":lower,
                        "exact_minus_lp_gap":worst-lower,"sound":int(lower<=worst+1e-5),
                        "candidate_source_count":len(source_nodes),
                        "potential_full_input_variables":int(local.size),
                        "selected_variables_json":json.dumps([[int(source_nodes[s]),f] for s,f in variables]),
                        "worst_flips_json":json.dumps([[int(source_nodes[s]),f] for s,f in worst_flips]),
                    })
    frame=pd.DataFrame(records); csv_path=output/"real-controlled-exact.csv"; frame.to_csv(csv_path,index=False)
    summary={"scope":"controlled subset only; not the full formal threat","cases":len(frame),
             "unique_real_targets":int(frame[["dataset","depth","target"]].drop_duplicates().shape[0]),
             "violations":int((frame.sound!=1).sum()),
             "max_lp_minus_exact":float((frame.lp_lower_bound-frame.controlled_exact_worst_margin).max()),
             "minimum_exact_minus_lp_gap":float(frame.exact_minus_lp_gap.min()),
             "attacks_per_case":sorted(int(value) for value in frame.controlled_attacks_evaluated.unique()),
             "csv":str(csv_path)}
    (output/"real-controlled-exact-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,sort_keys=True))
    if summary["violations"]: raise SystemExit(1)
    return summary


def parse(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--formal-root",required=True)
    parser.add_argument("--model-cache-dir",required=True); parser.add_argument("--output-dir",required=True)
    parser.add_argument("--cache-dir",default=".cache/pyg"); return parser.parse_args(argv)
if __name__=="__main__": run(parse())
