"""Family C unified runner: Many-objective NSGA-III / MOEA/D with per_silo and grouped modes.
PCA ignored (per 0.6). Uses Δ directly for grouping.
Supports objective reduction via similarity/dissimilarity/random groups, mean/min aggregation.
"""
import json, time, hashlib, dataclasses, platform
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from ..utils.seed import set_seed
from ..utils.config import ExperimentConfig
from ..data.datasets import make_synthetic_dataset, get_dataset, dirichlet_partition, heterogeneity_metrics, get_input_dim_and_n_classes
from ..clients.client import FederatedClient
from ..federated.characterization import characterize_clients
from ..federated.similarity import compute_similarity_matrix, compute_distance_matrix_from_similarity, similarity_stats
from ..federated.grouping import create_groups, grouping_summary
from ..evolution.many_objective import ManyObjectiveEvaluator
from ..evolution.grouped_many_evaluator import GroupedManyObjectiveEvaluator, GroupedManyProblem
from ..evolution.moead_runner import run_moead
from ..evolution.nsga3_runner import run_nsga3
from ..metrics.many_objective import compute_hv, compute_igd, get_nondominated, coverage_metric, find_knee_points
from ..models.genome import Genome

def _device(train_device: str):
    if train_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return train_device

def run_many_unified_experiment(cfg: ExperimentConfig,
                                optimizer: str = "nsga3",
                                objective_mode: str = "per_silo",  # per_silo | grouped
                                grouping_strategy: str = "similarity",  # similarity | dissimilarity | random
                                n_groups: int = None,  # e.g., N//2 or N//4 ; if None and grouped, default N//2
                                aggregation: str = "mean",  # mean | min
                                output_dir: Path = None,
                                verbose: bool = False):
    """
    Unified Family C runner.
    - per_silo: N objectives (N silos)
    - grouped: G objectives where G = n_groups (< N), each group aggregates members via mean/min.

    Groups built via S matrix (Δ direct, no PCA).
    """
    t_start=time.time()
    set_seed(cfg.seed)
    optimizer=optimizer.lower()
    assert optimizer in ("moead","nsga3","nsga-iii")
    if optimizer=="nsga-iii": optimizer="nsga3"
    assert objective_mode in ("per_silo","grouped")
    assert aggregation in ("mean","min")
    if output_dir is None:
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir=Path(cfg.output_dir)/f"many_{optimizer}_{objective_mode}_{ts}"
    output_dir=Path(output_dir)
    for sub in ["dataset","characterization","similarity","generations","plots","final_evaluation","pareto"]:
        (output_dir/sub).mkdir(parents=True, exist_ok=True)
    from ..utils.config import save_config
    save_config(cfg, output_dir/"config.json")
    try: save_config(cfg, output_dir/"config.yaml")
    except Exception: pass
    with open(output_dir/"run_meta.json","w") as f:
        json.dump({"optimizer":optimizer,"objective_mode":objective_mode,"grouping_strategy":grouping_strategy,"n_groups":n_groups,"aggregation":aggregation,"seed":cfg.seed,"n_clients":cfg.dataset.n_clients,"K":cfg.characterization.k_epochs,"alpha":cfg.dataset.alpha,"pop_size":cfg.evolution.pop_size,"n_generations":cfg.evolution.n_generations},f,indent=2)
    device=_device(cfg.train.device)
    # dataset
    t_data=time.time()
    if cfg.dataset.name=="synthetic":
        base_dataset=make_synthetic_dataset(n_samples=cfg.dataset.synthetic_n_samples,n_features=cfg.dataset.synthetic_n_features,n_informative=cfg.dataset.synthetic_n_informative,n_classes=cfg.dataset.synthetic_n_classes,seed=cfg.seed)
        labels=base_dataset.tensors[1].numpy()
    else:
        train_set=get_dataset(cfg.dataset.name, root=cfg.dataset.data_root, train=True, download=True)
        if hasattr(train_set,"targets"):
            labels=np.array(train_set.targets) if not isinstance(train_set.targets, torch.Tensor) else train_set.targets.numpy()
        else:
            labels=np.array([train_set[i][1] for i in range(len(train_set))])
        base_dataset=train_set
    input_dim,n_classes=get_input_dim_and_n_classes(base_dataset)
    partition=dirichlet_partition(labels=labels,n_clients=cfg.dataset.n_clients,alpha=cfg.dataset.alpha,seed=cfg.dataset.seed,min_samples_per_client=cfg.dataset.min_samples_per_client,val_ratio=cfg.dataset.val_ratio,test_ratio=cfg.dataset.test_ratio)
    clients={}
    for cid in range(cfg.dataset.n_clients):
        from torch.utils.data import Subset
        train_idxs=partition.splits[cid]["train"]; val_idxs=partition.splits[cid]["val"]; test_idxs=partition.splits[cid]["test"]
        train_ds=Subset(base_dataset,train_idxs); val_ds=Subset(base_dataset,val_idxs) if len(val_idxs)>0 else None
        test_ds=Subset(base_dataset,test_idxs) if len(test_idxs)>0 else None
        clients[cid]=FederatedClient(cid,train_ds,val_ds,test_ds,batch_size=cfg.train.batch_size,num_workers=cfg.train.num_workers)
    het=heterogeneity_metrics(partition.class_distribution)
    with open(output_dir/"dataset"/"heterogeneity.json","w") as f:
        json.dump({"partition":partition.to_dict(),"heterogeneity":het,"input_dim":input_dim,"n_classes":n_classes},f,indent=2,default=str)
    pd.DataFrame(partition.class_distribution, columns=[f"class_{i}" for i in range(partition.n_classes)]).to_csv(output_dir/"dataset"/"class_distribution.csv",index_label="client")
    data_time=time.time()-t_data
    # characterization
    t_char=time.time()
    char_res=characterize_clients(clients=clients,decoder_input_dim=input_dim,decoder_n_classes=n_classes,genome_config=dataclasses.asdict(cfg.genome),train_config=dataclasses.asdict(cfg.train),k_epochs=cfg.characterization.k_epochs,seed=cfg.characterization.seed,device=device)
    char_time=time.time()-t_char
    t_sim=time.time()
    S=compute_similarity_matrix(char_res["deltas"])
    D=compute_distance_matrix_from_similarity(S)
    stats_raw=similarity_stats(S)
    np.save(output_dir/"similarity"/"similarity_matrix.npy",S)
    np.save(output_dir/"similarity"/"distance_matrix.npy",D)
    pd.DataFrame(S).to_csv(output_dir/"similarity"/"similarity_matrix.csv",index=False)
    pd.DataFrame(D).to_csv(output_dir/"similarity"/"distance_matrix.csv",index=False)
    # also top-level for unified spec
    np.save(output_dir/"similarity_matrix.npy",S)
    np.save(output_dir/"distance_matrix.npy",D)
    pd.DataFrame(S).to_csv(output_dir/"similarity_matrix.csv",index=False)
    pd.DataFrame(D).to_csv(output_dir/"distance_matrix.csv",index=False)
    np.save(output_dir/"characterization"/"similarity_matrix.npy",S)
    np.save(output_dir/"characterization"/"distance_matrix.npy",D)
    delta_norms={str(cid):float(np.linalg.norm(v)) for cid,v in char_res["deltas"].items()}
    with open(output_dir/"similarity"/"delta_norms.json","w") as f: json.dump(delta_norms,f,indent=2)
    with open(output_dir/"similarity"/"similarity_stats.json","w") as f:
        json.dump({"similarity_stats":stats_raw,"n_params":char_res["n_params"],"k_epochs":cfg.characterization.k_epochs,"note":"No PCA, direct Δ"},f,indent=2)
    # true TV
    try:
        props=partition.class_distribution/partition.class_distribution.sum(axis=1,keepdims=True).clip(min=1)
        n=cfg.dataset.n_clients
        true_TV=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                true_TV[i,j]=np.abs(props[i]-props[j]).sum()/2
        np.save(output_dir/"similarity"/"true_TV_matrix.npy",true_TV)
    except Exception:
        pass
    t_sim_total=time.time()-t_sim

    # Determine grouping if needed
    groups=None
    grouping_info={}
    if objective_mode=="grouped":
        N=cfg.dataset.n_clients
        if n_groups is None:
            n_groups = N // 2
        # validate
        if n_groups > N:
            n_groups = N
        if N % n_groups != 0:
            # fallback to floor dimension: keep N//2 default if not divisible
            # try to make groups as balanced as possible via create_groups
            pass
        rng=np.random.RandomState(cfg.seed+777)
        groups=create_groups(S, n_groups=n_groups, strategy=grouping_strategy, rng=rng, D=D)
        grouping_info=grouping_summary(groups, S, D)
        grouping_info.update({"strategy":grouping_strategy,"aggregation":aggregation,"n_groups":n_groups,"objective_mode":objective_mode})
        with open(output_dir/"similarity"/"grouping.json","w") as f:
            json.dump(grouping_info,f,indent=2,default=str)
        with open(output_dir/"pairing.json","w") as f:
            json.dump({"objective_mode":objective_mode,"grouping_strategy":grouping_strategy,"n_groups":n_groups,"aggregation":aggregation,"groups":groups,"summary":grouping_info},f,indent=2)
        # also save as csv
        rows=[]
        for gid,g in enumerate(groups):
            for pos in g:
                rows.append({"group_id":gid,"pos":int(pos),"client_id":int(sorted(clients.keys())[pos]),"strategy":grouping_strategy,"aggregation":aggregation})
        pd.DataFrame(rows).to_csv(output_dir/"similarity"/"groups.csv",index=False)
    else:
        grouping_info={"objective_mode":"per_silo","n_groups":cfg.dataset.n_clients}
        with open(output_dir/"pairing.json","w") as f:
            json.dump(grouping_info,f,indent=2)
        groups=[[i] for i in range(cfg.dataset.n_clients)]

    # Evolution
    l_max=cfg.genome.l_max
    n_var=1+l_max+3
    # choose evaluator
    if objective_mode=="per_silo":
        evaluator=ManyObjectiveEvaluator(clients=clients,input_dim=input_dim,n_classes=n_classes,genome_config=dataclasses.asdict(cfg.genome),train_config=dataclasses.asdict(cfg.train),device=device,seed=cfg.evolution.seed,generation=0)
        n_obj = evaluator.n_obj
    else:
        evaluator=GroupedManyObjectiveEvaluator(clients=clients,input_dim=input_dim,n_classes=n_classes,genome_config=dataclasses.asdict(cfg.genome),train_config=dataclasses.asdict(cfg.train),groups=groups,aggregation=aggregation,device=device,seed=cfg.evolution.seed)
        n_obj = evaluator.n_obj

    t_evo=time.time()
    if optimizer=="moead":
        # for grouped, n_obj is G, handle reference dirs accordingly
        evo_res=run_moead(evaluator=evaluator,n_var=n_var,pop_size=cfg.evolution.pop_size,n_generations=cfg.evolution.n_generations,decomposition="tchebycheff",n_neighbors=min(20,cfg.evolution.pop_size),seed=cfg.evolution.seed,verbose=verbose)
    else:
        evo_res=run_nsga3(evaluator=evaluator,n_var=n_var,pop_size=cfg.evolution.pop_size,n_generations=cfg.evolution.n_generations,seed=cfg.evolution.seed,verbose=verbose)
    evo_time=time.time()-t_evo

    # Save histories
    hv_hist=evo_res["hv_history"]
    pd.DataFrame(hv_hist).to_csv(output_dir/"generations"/"hypervolume.csv",index=False)
    pd.DataFrame(hv_hist).to_csv(output_dir/"hypervolume.csv",index=False)
    # alias generation_metrics
    if hv_hist:
        pd.DataFrame(hv_hist).to_csv(output_dir/"generations"/"generation_metrics.csv",index=False)
        pd.DataFrame(hv_hist).to_csv(output_dir/"generation_metrics.csv",index=False)
    df_ind=pd.DataFrame(evaluator.generation_history)
    if not df_ind.empty:
        df_ind.to_csv(output_dir/"generations"/"population_history.csv",index=False)
        df_ind.to_csv(output_dir/"objective_vectors.csv",index=False)
        df_ind.to_csv(output_dir/"per_client_metrics.csv",index=False)
        # also generations/objective_vectors
        df_ind.to_csv(output_dir/"generations"/"objective_vectors.csv",index=False)
    else:
        pd.DataFrame().to_csv(output_dir/"objective_vectors.csv",index=False)

    # === Unified instrumentation for Family C ===
    # individual_node_accuracy.csv : each row per individual per node per generation
    try:
        ind_rows=[]
        # Identify best hashes for flags not needed now but we compute nondominated for final gen
        # For many-objective, is_nondominated true for final Pareto members
        # We need to determine final Pareto hashes: look at last generation's nondominated
        final_gen=cfg.evolution.n_generations
        final_sub=df_ind[df_ind["generation"]==final_gen] if not df_ind.empty else pd.DataFrame()
        # Get F matrix for final generation to compute nondominated mask
        final_nd_hashes=set()
        if not final_sub.empty:
            obj_cols=[c for c in final_sub.columns if c.startswith("obj_")]
            if obj_cols:
                F_final = final_sub[obj_cols].values
                try:
                    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                    nds=NonDominatedSorting().do(F_final, only_non_dominated_front=True)
                    # map to hashes
                    hashes=final_sub.iloc[nds]["genome_hash"].tolist() if len(nds)>0 else []
                    final_nd_hashes=set(hashes)
                except Exception:
                    final_nd_hashes=set()
        # Now build rows: for each rec in generation_history, we have acc_client_{cid} for all clients (since full eval per individual on all silos for many-objective)
        # Expand per node
        for _,rec in df_ind.iterrows():
            gen=int(rec["generation"])
            idx=int(rec["individual_idx"])
            gh=rec.get("genome_hash","")
            is_nd = gh in final_nd_hashes and gen==final_gen
            # is_knee, best_mean etc? leave false for now, fill later via pareto analysis
            for cid in sorted(clients.keys()):
                col_acc=f"acc_client_{cid}"
                col_loss=f"loss_client_{cid}"
                col_f1=f"f1_client_{cid}"
                if col_acc in rec:
                    ind_rows.append({
                        "seed":int(cfg.seed),
                        "generation":gen,
                        "individual_id":f"g{gen}_i{idx}",
                        "genome_hash":gh,
                        "node_id":int(cid),
                        "accuracy":float(rec[col_acc]),
                        "loss":float(rec.get(col_loss,0.0)),
                        "f1":float(rec.get(col_f1,0.0)),
                        "family":"many_objective",
                        "optimizer":optimizer,
                        "pairing_strategy": f"{objective_mode}_{grouping_strategy}_{aggregation}" if objective_mode=="grouped" else "per_silo",
                        "is_nondominated": bool(is_nd),
                        "is_knee": False,
                        "is_best_mean": False,
                        "is_best_worst": False,
                    })
        if ind_rows:
            df_ind_node=pd.DataFrame(ind_rows)
            # flag best_mean / best_worst per generation: find max mean_accuracy row
            # For each generation, best mean is individual with max mean_accuracy
            best_rows=df_ind.loc[df_ind.groupby("generation")["mean_accuracy"].idxmax()] if not df_ind.empty and "mean_accuracy" in df_ind.columns else pd.DataFrame()
            best_worst_rows=df_ind.loc[df_ind.groupby("generation")["worst_accuracy"].idxmax()] if not df_ind.empty and "worst_accuracy" in df_ind.columns else pd.DataFrame()
            best_hashes=set(best_rows["genome_hash"].tolist()) if not best_rows.empty else set()
            worst_hashes=set(best_worst_rows["genome_hash"].tolist()) if not best_worst_rows.empty else set()
            df_ind_node["is_best_mean"]=df_ind_node["genome_hash"].isin(best_hashes)
            df_ind_node["is_best_worst"]=df_ind_node["genome_hash"].isin(worst_hashes)
            # knee: determine from final pareto knee hashes (compute later but we have not yet computed pareto rows)
            df_ind_node.to_csv(output_dir/"individual_node_accuracy.csv",index=False)
            df_ind_node.to_csv(output_dir/"generations"/"individual_node_accuracy.csv",index=False)
            df_ind_node.to_csv(output_dir/"population_metrics.csv",index=False)
        # node_generation_summary
        if ind_rows:
            df_tmp=pd.DataFrame(ind_rows)
            sum_rows=[]
            for gen in sorted(df_tmp["generation"].unique()):
                sub_gen=df_tmp[df_tmp["generation"]==gen]
                for nid in sorted(df_tmp["node_id"].unique()):
                    vals=sub_gen[sub_gen["node_id"]==nid]["accuracy"].values
                    if len(vals)==0:
                        continue
                    sum_rows.append({
                        "generation":int(gen),"node_id":int(nid),
                        "best_accuracy":float(np.max(vals)),"mean_accuracy":float(np.mean(vals)),
                        "median_accuracy":float(np.median(vals)),"worst_accuracy":float(np.min(vals)),
                        "std_accuracy":float(np.std(vals)),"n_individuals":int(len(vals)),
                        "best_mean_model_accuracy": float(sub_gen["accuracy"].max()) if not sub_gen.empty else 0.0,
                    })
            if sum_rows:
                pd.DataFrame(sum_rows).to_csv(output_dir/"node_generation_summary.csv",index=False)
                pd.DataFrame(sum_rows).to_csv(output_dir/"generations"/"node_generation_summary.csv",index=False)
        # periodic full evaluation every 5 gens (already full eval per individual, but we can select best per gen and log separately)
        if not df_ind.empty and cfg.evolution.n_generations >=5:
            periodic_gens=list(range(5, cfg.evolution.n_generations+1, 5))
            if 1 not in periodic_gens:
                periodic_gens=[1]+periodic_gens
            periodic_rows=[]
            for gen in periodic_gens:
                sub=df_ind[df_ind["generation"]==gen]
                if sub.empty:
                    continue
                # best mean individual for that gen
                best_row=sub.loc[sub["mean_accuracy"].idxmax()]
                for cid in sorted(clients.keys()):
                    periodic_rows.append({
                        "generation":int(gen),
                        "genome_hash":best_row.get("genome_hash",""),
                        "node_id":int(cid),
                        "accuracy":float(best_row.get(f"acc_client_{cid}",0.0)),
                        "f1":float(best_row.get(f"f1_client_{cid}",0.0)),
                        "is_best_mean":True,
                    })
            if periodic_rows:
                pd.DataFrame(periodic_rows).to_csv(output_dir/"generations"/"periodic_full_evaluation.csv",index=False)
                pd.DataFrame(periodic_rows).to_csv(output_dir/"periodic_full_evaluation.csv",index=False)
    except Exception as e:
        print(f"[WARN] instrumentation many failed: {e}")
        import traceback; traceback.print_exc()

    # Pareto final
    res=evo_res["result"]
    F_final=res.pop.get("F") if res.pop is not None else np.array([])
    X_final=res.pop.get("X") if res.pop is not None else np.array([])
    hv_ref=evo_res["hv_ref"]
    if F_final is not None and len(F_final)>0:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds=NonDominatedSorting().do(F_final, only_non_dominated_front=True)
        pf_F=F_final[nds] if len(nds)>0 else F_final
        # Build pareto rows (reuse logic from runner_many but simplified)
        pareto_rows=[]
        for rank, idx in enumerate(nds):
            vec=X_final[idx]
            genome=Genome.from_vector(vec, dataclasses.asdict(cfg.genome))
            # For per_silo, accs from 1-F, for grouped need to recover group accs and per-client accs from df_ind?
            # Retrieve from evaluator history: find matching hash? Simpler use F to get group accs
            group_accs = 1 - F_final[idx]  # size G
            # For per-client accuracies, need to lookup df_ind final generation row matching hash
            # Try to find in df_ind
            gh=hashlib.md5(json.dumps(genome.to_dict(),sort_keys=True).encode()).hexdigest()[:12]
            # lookup per-client accs from df_ind final rows
            per_client_accs=[]
            df_match=df_ind[df_ind["genome_hash"]==gh] if not df_ind.empty else pd.DataFrame()
            if not df_match.empty:
                # take latest generation
                row=df_match.iloc[-1]
                per_client_accs=[float(row.get(f"acc_client_{cid}",0.0)) for cid in sorted(clients.keys())]
                mean_acc=float(row.get("mean_accuracy", np.mean(per_client_accs) if per_client_accs else 0.0))
                worst_acc=float(row.get("worst_accuracy", np.min(per_client_accs) if per_client_accs else 0.0))
                std_acc=float(row.get("std_accuracy",0.0))
                gap=float(row.get("gap",0.0))
            else:
                per_client_accs=[]
                mean_acc=float(np.mean(group_accs)) if len(group_accs)>0 else 0.0
                worst_acc=float(np.min(group_accs)) if len(group_accs)>0 else 0.0
                std_acc=float(np.std(group_accs)) if len(group_accs)>0 else 0.0
                gap=float(np.max(group_accs)-np.min(group_accs)) if len(group_accs)>0 else 0.0
            row_dict={
                "pareto_idx":int(rank),"pop_idx":int(idx),
                "genome":json.dumps(genome.to_dict()),
                "genome_hash":gh,
                "genome_L":genome.L,"genome_hidden":str(genome.hidden_sizes),"genome_act":genome.activation,
                "mean_accuracy":mean_acc,"worst_accuracy":worst_acc,"std_accuracy":std_acc,"gap":gap,
            }
            for g_idx, ga in enumerate(group_accs):
                row_dict[f"group_{g_idx}_acc"]=float(ga)
                row_dict[f"obj_{g_idx}"]=float(F_final[idx,g_idx])
            for cid_idx,cid in enumerate(sorted(clients.keys())):
                if per_client_accs:
                    row_dict[f"acc_client_{cid}"]=float(per_client_accs[cid_idx])
            pareto_rows.append(row_dict)
        df_pareto=pd.DataFrame(pareto_rows)
        df_pareto.to_csv(output_dir/"pareto"/"pareto_front.csv",index=False)
        df_pareto.to_csv(output_dir/"pareto_front.csv",index=False)
        df_pareto.to_csv(output_dir/"non_dominated.csv",index=False)
        # representatives: best mean, best worst, knee
        knee_indices=find_knee_points(F_final)
        # Save knee
        knee_rows=[]
        for k_idx in knee_indices[:3]:
            if k_idx < len(F_final):
                vec=X_final[k_idx]
                genome=Genome.from_vector(vec, dataclasses.asdict(cfg.genome))
                accs=1 - F_final[k_idx]
                knee_rows.append({"pop_idx":int(k_idx),"genome":json.dumps(genome.to_dict()),"mean_accuracy":float(np.mean(accs)),"worst_accuracy":float(np.min(accs))})
        if knee_rows:
            pd.DataFrame(knee_rows).to_csv(output_dir/"pareto"/"knee_points.csv",index=False)
        # hypervolume final
        hv_final=compute_hv(F_final, hv_ref)
        hv_pf=compute_hv(pf_F, hv_ref)
        hv_info={"hv_final_pop":float(hv_final),"hv_pareto":float(hv_pf),"hv_ref":hv_ref.tolist(),"n_pop":int(len(F_final)),"n_pareto":int(len(pf_F))}
    else:
        df_pareto=pd.DataFrame()
        hv_info={"hv_final_pop":0.0,"hv_pareto":0.0,"hv_ref":hv_ref.tolist() if "hv_ref" in locals() else [],"n_pop":0,"n_pareto":0}
        pd.DataFrame().to_csv(output_dir/"pareto_front.csv",index=False)
    with open(output_dir/"pareto"/"hypervolume.json","w") as f: json.dump(hv_info,f,indent=2)
    # per-client final evaluation for representatives (best mean/worst/knee)
    if not df_pareto.empty:
        reps={}
        best_mean_idx=int(df_pareto["mean_accuracy"].idxmax())
        reps["best_mean"]=df_pareto.iloc[best_mean_idx].to_dict()
        best_worst_idx=int(df_pareto["worst_accuracy"].idxmax())
        reps["best_worst"]=df_pareto.iloc[best_worst_idx].to_dict()
        if 'knee_indices' in locals() and knee_indices:
            k_idx=knee_indices[0]
            reps["knee"]=df_pareto[df_pareto["pop_idx"]==k_idx].iloc[0].to_dict() if not df_pareto[df_pareto["pop_idx"]==k_idx].empty else df_pareto.iloc[best_mean_idx].to_dict()
        with open(output_dir/"pareto"/"representatives.json","w") as f:
            json.dump(reps,f,indent=2,default=str)
        rep_rows=[]
        for key in ["best_mean","best_worst","knee"]:
            if key in reps:
                r=dict(reps[key]); r["type"]=key
                rep_rows.append(r)
        if rep_rows:
            pd.DataFrame(rep_rows).to_csv(output_dir/"final_evaluation"/"representatives_metrics.csv",index=False)
            pd.DataFrame(rep_rows).to_csv(output_dir/"final_full_evaluation.csv",index=False)
            pd.DataFrame(rep_rows).to_csv(output_dir/"per_client_metrics.csv",index=False)
    # runtime/cost
    total_time=time.time()-t_start
    cost_info={
        "total_time":float(total_time),"data_time":float(data_time),"char_time":float(char_time),"sim_time":float(t_sim_total),"evo_time":float(evo_time),
        "eval_count":int(evaluator.eval_count),"n_obj":int(n_obj),"n_clients":int(cfg.dataset.n_clients),
        "pop_size_actual":int(evo_res.get("pop_size_actual",cfg.evolution.pop_size)),"n_generations":int(cfg.evolution.n_generations),
        "total_evals_possible_full":int(cfg.evolution.pop_size*cfg.evolution.n_generations*cfg.dataset.n_clients),
        "evaluations":int(evaluator.eval_count),"optimizer":optimizer,
        "objective_mode":objective_mode,"grouping_strategy":grouping_strategy,"aggregation":aggregation,"n_groups":n_groups if n_groups else cfg.dataset.n_clients,
    }
    with open(output_dir/"runtime.json","w") as f: json.dump(cost_info,f,indent=2)
    with open(output_dir/"cost.json","w") as f: json.dump(cost_info,f,indent=2)
    with open(output_dir/"metrics.json","w") as f:
        json.dump({"hv":hv_info,"cost":cost_info,"heterogeneity":het,"grouping":grouping_info},f,indent=2)
    with open(output_dir/"metadata.json","w") as f:
        json.dump({"optimizer":optimizer,"objective_mode":objective_mode,"grouping_strategy":grouping_strategy,"aggregation":aggregation,"dataset":dataclasses.asdict(cfg.dataset),"train":dataclasses.asdict(cfg.train),"genome":dataclasses.asdict(cfg.genome),"evolution":{**dataclasses.asdict(cfg.evolution),"pop_size_actual":cost_info["pop_size_actual"]},"hv_ref":hv_ref.tolist(),"device":device,"seed":cfg.seed,"total_time":total_time,"platform":platform.platform(),"python":platform.python_version()},f,indent=2,default=str)
    # also save config copy top-level
    # generation_metrics already done etc.
    # Ensure files required by spec exist: population_metrics.csv alias already, objective_vectors, etc.
    # Create placeholder if missing
    for required in ["generation_metrics.csv","individual_node_accuracy.csv","node_generation_summary.csv","population_metrics.csv","objective_vectors.csv","pairing.json","similarity_matrix.csv","distance_matrix.csv","final_full_evaluation.csv","pareto_front.csv"]:
        if not (output_dir/required).exists():
            # try to create empty or copy from alternative
            alt = output_dir/"generations"/required
            if alt.exists():
                try:
                    pd.read_csv(alt).to_csv(output_dir/required,index=False)
                except Exception:
                    pass
            else:
                pd.DataFrame().to_csv(output_dir/required,index=False)
    return {"output_dir":output_dir,"hv_history":hv_hist,"evaluator":evaluator,"cost":cost_info,"hv_info":hv_info,"grouping_info":grouping_info}
