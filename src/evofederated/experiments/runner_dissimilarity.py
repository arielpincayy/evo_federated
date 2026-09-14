"""Family A runner: Dissimilarity pairing + NSGA-II per pair."""
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
from ..federated.matching import max_dissimilarity_matching, random_pairing, probabilistic_dissimilarity_pairing, matching_summary, get_representatives
from ..evolution.dissimilarity_evaluator import PairEvaluator, DissimilarityPairProblem
from ..evolution.nsga2 import run_nsga2
from ..models.genome import Genome
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

def _device(train_device: str):
    if train_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return train_device

def _run_one_pair(cfg: ExperimentConfig, clients, input_dim, n_classes, pair_idx, client_a, client_b, S, D, pairing_strategy, tau, output_dir: Path, device, verbose=False):
    """Run NSGA-II for a single pair."""
    output_dir = Path(output_dir)
    pair_dir = output_dir / f"pair_{pair_idx:02d}_{client_a}_{client_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    evaluator = PairEvaluator(
        clients=clients, client_a_id=client_a, client_b_id=client_b,
        input_dim=input_dim, n_classes=n_classes,
        genome_config=dataclasses.asdict(cfg.genome),
        train_config=dataclasses.asdict(cfg.train),
        pair_id=pair_idx, pair_tuple=(client_a, client_b),
        device=device, seed=cfg.evolution.seed,
        family="dissimilarity", pairing_strategy=pairing_strategy
    )
    l_max = cfg.genome.l_max
    n_var = 1 + l_max + 3
    hv_ref = np.array([1.1, 1.1])  # since objectives are 1-acc in [0,1], ref worse =1.1
    # custom run_nsga2 call: we need to set hv_ref appropriately; run_nsga2 expects tuple
    # We'll call minimize directly here to avoid hyperparam confusion, but reuse run_nsga2 logic
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.core.callback import Callback
    hv_history=[]
    start_time=time.time()
    class PairCallback(Callback):
        def notify(self, algorithm):
            cur_gen = int(algorithm.n_gen)
            evaluator.set_generation(cur_gen)
            F = algorithm.pop.get("F")
            if F is not None and len(F)>0:
                try:
                    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                    pf = F[nds]
                except Exception:
                    pf = F
                try:
                    hv = HV(ref_point=hv_ref)
                    hv_val = hv.do(pf) if len(pf) else 0.0
                except Exception:
                    hv_val=0.0
                n_nd = int(len(pf)) if pf is not None else 0
            else:
                hv_val=0.0; n_nd=0
            elapsed=time.time()-start_time
            recs=[r for r in evaluator.generation_history if r["generation"]==cur_gen]
            if recs:
                mean_acc = float(np.mean([r["mean_accuracy"] for r in recs]))
                worst_acc = float(np.mean([r["min_accuracy"] for r in recs]))
            else:
                mean_acc=worst_acc=0.0
            hv_history.append({"generation":cur_gen,"hv":float(hv_val),"n_nd":n_nd,"mean_accuracy":mean_acc,"worst_accuracy":worst_acc,"eval_count":int(evaluator.eval_count),"elapsed":float(elapsed)})

    algorithm = NSGA2(pop_size=cfg.evolution.pop_size,
                      sampling=FloatRandomSampling(),
                      crossover=SBX(prob=cfg.evolution.crossover_prob, eta=cfg.evolution.crossover_eta),
                      mutation=PM(eta=cfg.evolution.mutation_eta, prob=cfg.evolution.mutation_prob),
                      eliminate_duplicates=True)
    problem = DissimilarityPairProblem(n_var=n_var, evaluator=evaluator)
    callback = PairCallback()
    res = minimize(problem, algorithm, ("n_gen", cfg.evolution.n_generations), seed=cfg.evolution.seed, callback=callback, verbose=verbose)

    # Save histories
    pd.DataFrame(hv_history).to_csv(pair_dir / "hypervolume.csv", index=False)
    pd.DataFrame(evaluator.generation_history).to_csv(pair_dir / "population_history.csv", index=False)
    pd.DataFrame(evaluator.generation_history).to_csv(pair_dir / "individual_node_accuracy_pair.csv", index=False)

    # Final population
    F_final = res.pop.get("F") if res.pop is not None else np.array([])
    X_final = res.pop.get("X") if res.pop is not None else np.array([])
    if F_final is not None and len(F_final)>0:
        nds = NonDominatedSorting().do(F_final, only_non_dominated_front=True)
        is_nd = np.zeros(len(F_final), dtype=bool); is_nd[nds]=True
        pf_F = F_final[nds] if len(nds)>0 else F_final
        pf_X = X_final[nds] if len(nds)>0 else X_final
        try:
            hv = HV(ref_point=hv_ref); hv_val = hv.do(pf_F) if len(pf_F) else 0.0
        except Exception:
            hv_val=0.0
        # Build pareto dataframe
        pareto_rows=[]
        for idx in nds:
            vec = X_final[idx]
            genome = Genome.from_vector(vec, dataclasses.asdict(cfg.genome))
            acc_a = 1 - F_final[idx,0]; acc_b = 1 - F_final[idx,1]
            row = {
                "pareto_idx": int(idx), "pop_idx": int(idx),
                "genome": json.dumps(genome.to_dict()),
                "genome_hash": hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()[:12],
                "genome_L": genome.L, "genome_hidden": str(genome.hidden_sizes),
                "genome_act": genome.activation,
                "accuracy_a": float(acc_a), "accuracy_b": float(acc_b),
                "mean_accuracy": float((acc_a+acc_b)/2), "min_accuracy": float(min(acc_a, acc_b)),
                "max_accuracy": float(max(acc_a, acc_b)), "gap": float(abs(acc_a-acc_b)),
                "obj_a": float(F_final[idx,0]), "obj_b": float(F_final[idx,1]),
                "is_nondominated": True,
            }
            pareto_rows.append(row)
        df_pareto = pd.DataFrame(pareto_rows)
        df_pareto.to_csv(pair_dir / "pair_pareto_front.csv", index=False)
        df_pareto.to_csv(pair_dir / "pareto_front.csv", index=False)
        # per pair correlation analysis: accuracy correlation between objectives
        # Compute correlation between acc_a and acc_b across population
        if len(F_final)>2:
            acc_a_all = 1 - F_final[:,0]; acc_b_all = 1 - F_final[:,1]
            try:
                corr = float(np.corrcoef(acc_a_all, acc_b_all)[0,1])
            except Exception:
                corr=0.0
        else:
            corr=0.0
    else:
        df_pareto=pd.DataFrame(); hv_val=0.0; corr=0.0; pareto_rows=[]

    # Determine representative models for full evaluation across all nodes
    # Use endpoints, knee, best min, best mean, min imbalance
    rep_genomes = []
    if pareto_rows:
        df_tmp = pd.DataFrame(pareto_rows)
        # best mean
        best_mean_idx = int(df_tmp["mean_accuracy"].idxmax())
        rep_genomes.append(("best_mean", pareto_rows[best_mean_idx]))
        # best worst (max min)
        best_min_idx = int(df_tmp["min_accuracy"].idxmax())
        rep_genomes.append(("best_min", pareto_rows[best_min_idx]))
        # extremes
        max_a_idx = int(df_tmp["accuracy_a"].idxmax())
        rep_genomes.append(("extreme_a", pareto_rows[max_a_idx]))
        max_b_idx = int(df_tmp["accuracy_b"].idxmax())
        rep_genomes.append(("extreme_b", pareto_rows[max_b_idx]))
        # knee: point farthest from line connecting extremes
        if len(df_tmp)>=3:
            # normalized objectives
            f_min = np.array([df_tmp["obj_a"].min(), df_tmp["obj_b"].min()])
            f_max = np.array([df_tmp["obj_a"].max(), df_tmp["obj_b"].max()])
            denom = (f_max - f_min).clip(min=1e-9)
            pf_norm = np.stack([(df_tmp["obj_a"]-f_min[0])/denom[0], (df_tmp["obj_b"]-f_min[1])/denom[1]], axis=1)
            # line between extremes
            idx_min0 = int(np.argmin(pf_norm[:,0])); idx_min1 = np.argmin(pf_norm[:,1])
            p0=pf_norm[idx_min0]; p1=pf_norm[idx_min1]
            line_vec = p1-p0; ll=np.linalg.norm(line_vec)
            if ll>1e-9:
                cross=np.abs((pf_norm[:,0]-p0[0])*line_vec[1] - (pf_norm[:,1]-p0[1])*line_vec[0])
                dists=cross/ll
                knee_idx=int(np.argmax(dists))
                rep_genomes.append(("knee", pareto_rows[knee_idx]))
            else:
                rep_genomes.append(("knee", pareto_rows[best_mean_idx]))
        # min imbalance
        min_gap_idx = int(df_tmp["gap"].idxmin())
        rep_genomes.append(("min_imbalance", pareto_rows[min_gap_idx]))
        # dedup by hash
        seen=set(); uniq=[]
        for typ,row in rep_genomes:
            h=row["genome_hash"]
            if h not in seen:
                seen.add(h)
                r=dict(row); r["type"]=typ
                uniq.append(r)
        rep_genomes=uniq
    else:
        rep_genomes=[]

    # Full evaluation across all clients for representative models
    all_client_ids = sorted(clients.keys())
    full_rows=[]
    for prow in rep_genomes:
        typ = prow.get("type","unknown")
        genome = Genome.from_dict(json.loads(prow["genome"]))
        # evaluate on all clients
        # use evaluator's full method but need to reconstruct detailed per-client accuracy
        from ..models.genome import genome_to_model
        from ..federated.characterization import flatten_params, set_model_params
        seed = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
        seed_int = int(seed[:8],16) % (2**31-1)
        torch.manual_seed(seed_int)
        template = genome_to_model(genome, input_dim, n_classes)
        flat_init = flatten_params(template)
        accs=[]
        f1s=[]; losses=[]
        per_client={}
        for cid in all_client_ids:
            model = genome_to_model(genome, input_dim, n_classes)
            set_model_params(model, flat_init)
            client = clients[cid]
            train_res = client.train(model, epochs=cfg.train.epochs, lr=cfg.train.lr, optimizer_name=cfg.train.optimizer, device=device)
            eval_res = client.evaluate(model, device=device, split="test")
            accs.append(float(eval_res["accuracy"])); f1s.append(float(eval_res["macro_f1"])); losses.append(float(eval_res["loss"]))
            per_client[int(cid)] = float(eval_res["accuracy"])
        row={
            "pair_id": int(pair_idx), "type": typ, "genome_hash": prow["genome_hash"],
            "genome": prow["genome"],
            "pair_accuracy_a": float(prow["accuracy_a"]), "pair_accuracy_b": float(prow["accuracy_b"]),
            "pair_mean": float(prow["mean_accuracy"]),
            "global_mean_accuracy": float(np.mean(accs)), "global_median": float(np.median(accs)),
            "global_worst": float(np.min(accs)), "global_best": float(np.max(accs)),
            "global_std": float(np.std(accs)), "global_gap": float(np.max(accs)-np.min(accs)),
            "n_clients": int(len(all_client_ids)),
        }
        for cid_idx,cid in enumerate(all_client_ids):
            row[f"acc_client_{cid}"]=float(accs[cid_idx])
            row[f"f1_client_{cid}"]=float(f1s[cid_idx])
        full_rows.append(row)
    if full_rows:
        pd.DataFrame(full_rows).to_csv(pair_dir / "final_full_evaluation.csv", index=False)

    return {
        "pair_idx": pair_idx,
        "client_a": client_a, "client_b": client_b,
        "evaluator": evaluator,
        "hv_history": hv_history,
        "hv_final": float(hv_val) if 'hv_val' in locals() else 0.0,
        "corr": float(corr) if 'corr' in locals() else 0.0,
        "df_pareto": df_pareto if 'df_pareto' in locals() else pd.DataFrame(),
        "full_rows": full_rows,
        "rep_genomes": rep_genomes,
    }


def run_dissimilarity_experiment(cfg: ExperimentConfig, output_dir: Path=None, verbose=False,
                                 pairing_mode="deterministic", pairing_tau=None):
    """
    pairing_mode: deterministic | random | probabilistic
    For probabilistic, pairing_tau required.
    Returns summary.
    """
    t_start=time.time()
    set_seed(cfg.seed)
    if output_dir is None:
        from datetime import datetime
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir=Path(cfg.output_dir)/f"dissimilarity_{ts}"
    output_dir=Path(output_dir)
    for sub in ["dataset","characterization","similarity","generations","plots","final_evaluation","pairs"]:
        (output_dir/sub).mkdir(parents=True, exist_ok=True)
    from ..utils.config import save_config
    save_config(cfg, output_dir / "config.json")
    try: save_config(cfg, output_dir / "config.yaml")
    except Exception: pass
    device=_device(cfg.train.device)
    # dataset
    t_data=time.time()
    if cfg.dataset.name=="synthetic":
        base_dataset=make_synthetic_dataset(n_samples=cfg.dataset.synthetic_n_samples,
                                            n_features=cfg.dataset.synthetic_n_features,
                                            n_informative=cfg.dataset.synthetic_n_informative,
                                            n_classes=cfg.dataset.synthetic_n_classes,
                                            seed=cfg.seed)
        labels=base_dataset.tensors[1].numpy()
    else:
        train_set=get_dataset(cfg.dataset.name, root=cfg.dataset.data_root, train=True, download=True)
        if hasattr(train_set,"targets"):
            labels=np.array(train_set.targets) if not isinstance(train_set.targets, torch.Tensor) else train_set.targets.numpy()
        else:
            labels=np.array([train_set[i][1] for i in range(len(train_set))])
        base_dataset=train_set
    input_dim,n_classes=get_input_dim_and_n_classes(base_dataset)
    partition=dirichlet_partition(labels=labels,n_clients=cfg.dataset.n_clients,alpha=cfg.dataset.alpha,
                                  seed=cfg.dataset.seed,min_samples_per_client=cfg.dataset.min_samples_per_client,
                                  val_ratio=cfg.dataset.val_ratio,test_ratio=cfg.dataset.test_ratio)
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
    pd.DataFrame(partition.class_distribution, columns=[f"class_{i}" for i in range(partition.n_classes)]).to_csv(output_dir/"dataset"/"class_distribution.csv", index_label="client")
    data_time=time.time()-t_data

    # characterization
    t_char=time.time()
    char_res=characterize_clients(clients=clients, decoder_input_dim=input_dim, decoder_n_classes=n_classes,
                                  genome_config=dataclasses.asdict(cfg.genome), train_config=dataclasses.asdict(cfg.train),
                                  k_epochs=cfg.characterization.k_epochs, seed=cfg.characterization.seed, device=device)
    char_time=time.time()-t_char
    S=compute_similarity_matrix(char_res["deltas"])
    D=compute_distance_matrix_from_similarity(S)
    sim_stats=similarity_stats(S)
    from ..federated.divergence import divergence_stats
    div_stats=divergence_stats(D)
    np.save(output_dir/"similarity"/"similarity_matrix.npy",S)
    np.save(output_dir/"similarity"/"distance_matrix.npy",D)
    np.save(output_dir/"characterization"/"similarity_matrix.npy",S)
    np.save(output_dir/"characterization"/"divergence_matrix.npy",D)
    # also divergence alias for compatibility
    pd.DataFrame(S).to_csv(output_dir/"similarity"/"similarity_matrix.csv", index=False)
    pd.DataFrame(D).to_csv(output_dir/"similarity"/"distance_matrix.csv", index=False)
    pd.DataFrame(S).to_csv(output_dir/"characterization"/"similarity_matrix.csv", index=False)
    pd.DataFrame(D).to_csv(output_dir/"characterization"/"divergence_matrix.csv", index=False)
    delta_norms={str(cid):float(np.linalg.norm(v)) for cid,v in char_res["deltas"].items()}
    with open(output_dir/"similarity"/"delta_norms.json","w") as f:
        json.dump(delta_norms,f,indent=2)
    with open(output_dir/"similarity"/"similarity_stats.json","w") as f:
        json.dump({"similarity_stats":sim_stats,"distance_stats":div_stats,"probe_genome":char_res["probe_genome"].to_dict(),"n_params":char_res["n_params"],"k_epochs":cfg.characterization.k_epochs},f,indent=2,default=str)
    # true TV for correlation
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

    # pairing
    pairing_rng=np.random.RandomState(cfg.seed+301)
    prob_info={}
    if pairing_mode=="random":
        pairs,singletons=random_pairing(S, pairing_rng)
        summary=matching_summary(S,pairs,singletons,D)
        summary["pairing_mode"]="random"
        strategy_label="random"
    elif pairing_mode=="probabilistic":
        tau=float(pairing_tau) if pairing_tau is not None else 0.5
        pairs,singletons,prob_info=probabilistic_dissimilarity_pairing(S,tau,pairing_rng,D)
        summary=matching_summary(S,pairs,singletons,D)
        summary["pairing_mode"]="probabilistic"
        summary["tau"]=tau
        summary.update(prob_info)
        strategy_label=f"prob_tau_{tau}"
    else:
        pairs,singletons=max_dissimilarity_matching(S,D)
        summary=matching_summary(S,pairs,singletons,D)
        summary["pairing_mode"]="deterministic"
        strategy_label="deterministic_dissimilarity"

    client_ids_sorted=sorted(clients.keys())
    pairs_cids=[(client_ids_sorted[a],client_ids_sorted[b]) for a,b in pairs]
    singletons_cids=[client_ids_sorted[s] for s in singletons]
    with open(output_dir/"similarity"/"pairs.json","w") as f:
        json.dump({"pairs_pos":pairs,"pairs_client_ids":pairs_cids,"singletons_pos":singletons,"singletons_client_ids":singletons_cids,"summary":summary,"pairing_mode":pairing_mode,"pairing_tau":pairing_tau},f,indent=2,default=str)
    pair_rows=[]
    for idx,(a,b) in enumerate(pairs):
        pair_rows.append({"pair_id":idx,"client_a_pos":int(a),"client_b_pos":int(b),"client_a_id":int(client_ids_sorted[a]),"client_b_id":int(client_ids_sorted[b]),"similarity":float(S[a,b]),"distance":float(D[a,b])})
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(output_dir/"similarity"/"pairs.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(output_dir/"pairs"/"pairs.csv", index=False)

    # NSGA-II per pair
    t_evo=time.time()
    pair_results=[]
    total_evals=0
    for idx,(pos_a,pos_b) in enumerate(pairs):
        client_a=client_ids_sorted[pos_a]; client_b=client_ids_sorted[pos_b]
        res=_run_one_pair(cfg,clients,input_dim,n_classes,idx,client_a,client_b,S,D,pairing_mode if pairing_mode!="probabilistic" else f"prob_dissim_tau_{pairing_tau}", pairing_tau, output_dir, device, verbose)
        pair_results.append(res)
        total_evals+=res["evaluator"].eval_count

    # handle singleton (if odd N, one client remains unpaired – treat as single objective? but spec says pairs of 2, singleton evaluated alone? For NSGA-II we need 2 objectives, so singletons cannot be optimized with 2 objectives. We'll evaluate singleton as single-client baseline but not via NSGA-II 2-obj; we log it.)
    singleton_rows=[]
    for s_pos in singletons:
        cid=client_ids_sorted[s_pos]
        # For singleton, we could run a simple single-objective GA on that client alone? But spec says each pair independent 2-obj. Singletons are just leftover; we'll note them.
        singleton_rows.append({"singleton_pos":int(s_pos),"singleton_id":int(cid)})

    evo_time=time.time()-t_evo
    total_time=time.time()-t_start

    # Aggregate per-pair metrics
    hv_list=[r["hv_final"] for r in pair_results]
    corr_list=[r["corr"] for r in pair_results]
    # Save aggregate CSVs
    agg_rows=[]
    for r in pair_results:
        agg_rows.append({"pair_id":r["pair_idx"],"client_a":r["client_a"],"client_b":r["client_b"],"hv":r["hv_final"],"corr":r["corr"],"n_pareto":len(r["df_pareto"]) if not r["df_pareto"].empty else 0})
    pd.DataFrame(agg_rows).to_csv(output_dir/"pair_aggregate.csv", index=False)

    # Build individual_node_accuracy.csv across all pairs
    # For each evaluator's generation_history, we have per individual 2 accuracies. Expand to two rows per individual (one per node)
    ind_rows=[]
    for r in pair_results:
        ev=r["evaluator"]
        for rec in ev.generation_history:
            # two rows per individual
            base={"seed":int(cfg.seed),"generation":int(rec["generation"]),"individual_id":f"pair{r['pair_idx']}_i{rec['individual_idx']}",
                  "family":"dissimilarity","optimizer":"nsga2","pairing_strategy":pairing_mode if pairing_mode!="probabilistic" else f"prob_tau_{pairing_tau}",
                  "pair_id":int(rec["pair_id"]), "pair":rec["pair"]}
            # need is_nondominated? For final gen we can compute, but for intermediate we approximate via per-gen nondominated?
            # For simplicity mark is_nondominated false except for final Pareto members; we could compute per generation but expensive; set false and later update final gen
            for cid,acc,loss,f1 in [(rec["client_a"], rec["accuracy_a"], rec["loss_a"], rec["f1_a"]),(rec["client_b"], rec["accuracy_b"], rec["loss_b"], rec["f1_b"])]:
                ind_rows.append({**base,"node_id":int(cid),"accuracy":float(acc),"loss":float(loss),"f1":float(f1),"is_nondominated":False,"is_knee":False,"is_best_mean":False,"is_best_worst":False})
            # To know nondominated per generation, we compute after loop? Could update later.
        # update final generation nondominated flags by checking df_pareto
        if not r["df_pareto"].empty and r["evaluator"].generation_history:
            # final gen = n_generations
            final_gen=cfg.evolution.n_generations
            # hashes of pareto members
            pareto_hashes=set(r["df_pareto"]["genome_hash"].tolist())
            for row in ind_rows:
                if row["generation"]==final_gen and row["pair_id"]==r["pair_idx"]:
                    # need to map individual hash? rec stores genome but row lost hash; we could store hash in rec
                    # Instead iterate evaluator history final gen and mark
                    pass

    # For improved marking, recompute per generation nondominated? We'll keep simple for now: per pair final Pareto hashes
    # Rebuild with hashes
    ind_rows=[]
    for r in pair_results:
        ev=r["evaluator"]
        pareto_hashes=set(r["df_pareto"]["genome_hash"].tolist()) if not r["df_pareto"].empty else set()
        # knee hashes if available? we have rep_genomes knee hashes in full_rows? we can compute knee from pareto rows but simplified
        knee_hashes=set()
        # maybe identify knee as row with type knee from full_rows? Actually rep_genomes includes knee
        # We'll attempt to identify knee hash via smallest gap? not needed now
        for rec in ev.generation_history:
            gh=rec["genome_hash"]
            is_nd= gh in pareto_hashes and rec["generation"]==cfg.evolution.n_generations
            base={"seed":int(cfg.seed),"generation":int(rec["generation"]),"individual_id":f"pair{r['pair_idx']}_i{rec['individual_idx']}",
                  "genome_hash": gh, "family":"dissimilarity","optimizer":"nsga2","pairing_strategy":pairing_mode if pairing_mode!="probabilistic" else f"prob_tau_{pairing_tau}",
                  "pair_id":int(rec["pair_id"]), "pair":rec["pair"],
                  "is_nondominated": bool(is_nd),"is_knee":False,"is_best_mean":False,"is_best_worst":False}
            for cid,acc,loss,f1 in [(rec["client_a"], rec["accuracy_a"], rec["loss_a"], rec["f1_a"]),(rec["client_b"], rec["accuracy_b"], rec["loss_b"], rec["f1_b"])]:
                ind_rows.append({**base,"node_id":int(cid),"accuracy":float(acc),"loss":float(loss),"f1":float(f1)})
    pd.DataFrame(ind_rows).to_csv(output_dir/"individual_node_accuracy.csv", index=False)
    pd.DataFrame(ind_rows).to_csv(output_dir/"generations"/"individual_node_accuracy.csv", index=False)

    # node_generation_summary.csv : per generation per node best/mean/median/worst across all individuals evaluating that node (only pairs containing node)
    # For each generation and each node, collect accuracies where node_id==that node
    if ind_rows:
        df_ind=pd.DataFrame(ind_rows)
        summary_rows=[]
        for gen in sorted(df_ind["generation"].unique()):
            sub_gen=df_ind[df_ind["generation"]==gen]
            for nid in sorted(df_ind["node_id"].unique()):
                vals=sub_gen[sub_gen["node_id"]==nid]["accuracy"].values
                if len(vals)==0:
                    continue
                summary_rows.append({
                    "generation":int(gen),"node_id":int(nid),
                    "best_accuracy":float(np.max(vals)),"mean_accuracy":float(np.mean(vals)),
                    "median_accuracy":float(np.median(vals)),"worst_accuracy":float(np.min(vals)),
                    "std_accuracy":float(np.std(vals)),"n_individuals":int(len(vals)),
                })
        pd.DataFrame(summary_rows).to_csv(output_dir/"node_generation_summary.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(output_dir/"generations"/"node_generation_summary.csv", index=False)
    else:
        summary_rows=[]

    # final_full_evaluation.csv aggregated across pairs representatives
    all_full=[]
    for r in pair_results:
        all_full.extend(r["full_rows"])
    if all_full:
        pd.DataFrame(all_full).to_csv(output_dir/"final_full_evaluation.csv", index=False)
        pd.DataFrame(all_full).to_csv(output_dir/"final_evaluation"/"final_full_evaluation.csv", index=False)

    # cost
    theoretical_full=cfg.evolution.pop_size*cfg.dataset.n_clients*cfg.evolution.n_generations
    # Family A evaluations: per pair pop *2 *gens ; total = n_pairs * pop*2 *gens
    total_search=total_evals
    cost={
        "n_clients":int(cfg.dataset.n_clients),"n_pairs":int(len(pairs)),"n_singletons":int(len(singletons)),
        "pop_size":int(cfg.evolution.pop_size),"n_generations":int(cfg.evolution.n_generations),
        "total_search_evaluations":int(total_search),"theoretical_full_evaluations":int(theoretical_full),
        "evaluations_per_generation":int(cfg.evolution.pop_size*2*len(pairs)) if pairs else 0,
        "characterization_evaluations":int(cfg.dataset.n_clients),"characterization_time":float(char_time),
        "evolution_time":float(evo_time),"total_time":float(total_time),
        "pairing_mode":pairing_mode,"tau":pairing_tau,
    }
    with open(output_dir/"cost.json","w") as f: json.dump(cost,f,indent=2)
    with open(output_dir/"runtime.json","w") as f: json.dump(cost,f,indent=2)
    with open(output_dir/"similarity"/"cost.json","w") as f: json.dump(cost,f,indent=2)

    # generation_metrics.csv per pair hv aggregated? Create per generation avg
    # For family A, hv per generation we have per pair; aggregate mean hv
    all_hv=[]
    for r in pair_results:
        for rec in r["hv_history"]:
            all_hv.append({"generation":rec["generation"],"pair_id":r["pair_idx"],"hv":rec["hv"],"n_nd":rec["n_nd"],"mean_accuracy":rec["mean_accuracy"]})
    if all_hv:
        pd.DataFrame(all_hv).to_csv(output_dir/"generations"/"generation_metrics.csv", index=False)
        pd.DataFrame(all_hv).to_csv(output_dir/"generation_metrics.csv", index=False)
        # also hv_history.csv alias
        pd.DataFrame(all_hv).to_csv(output_dir/"generations"/"hypervolume_history.csv", index=False)
    # population_metrics.csv alias
    if ind_rows:
        pd.DataFrame(ind_rows).to_csv(output_dir/"population_metrics.csv", index=False)

    # objective_vectors.csv : per individual F vectors
    obj_rows=[]
    for r in pair_results:
        ev=r["evaluator"]
        for rec in ev.generation_history:
            obj_rows.append({
                "generation":rec["generation"],"pair_id":rec["pair_id"],"individual_idx":rec["individual_idx"],
                "obj_a":rec["obj_a"],"obj_b":rec["obj_b"],
                "accuracy_a":rec["accuracy_a"],"accuracy_b":rec["accuracy_b"],
                "mean_accuracy":rec["mean_accuracy"],"min_accuracy":rec["min_accuracy"]
            })
    if obj_rows:
        pd.DataFrame(obj_rows).to_csv(output_dir/"objective_vectors.csv", index=False)

    # config and pairing copies
    with open(output_dir/"pairing.json","w") as f:
        json.dump({"pairs_pos":pairs,"pairs_client_ids":pairs_cids,"singletons_pos":singletons,"singletons_client_ids":singletons_cids,"summary":summary,"pairing_mode":pairing_mode,"pairing_tau":pairing_tau},f,indent=2,default=str)
    np.save(output_dir/"similarity_matrix.npy",S)
    pd.DataFrame(S).to_csv(output_dir/"similarity_matrix.csv", index=False)
    pd.DataFrame(D).to_csv(output_dir/"distance_matrix.csv", index=False)
    # also copy to characterization
    np.save(output_dir/"characterization"/"similarity_matrix.npy",S)
    np.save(output_dir/"characterization"/"distance_matrix.npy",D)

    summary_dict={
        "seed":int(cfg.seed),"pairing_mode":pairing_mode,"pairing_tau":pairing_tau,
        "n_pairs":int(len(pairs)),"n_clients":int(cfg.dataset.n_clients),
        "mean_hv":float(np.mean(hv_list)) if hv_list else 0.0,
        "mean_corr":float(np.mean(corr_list)) if corr_list else 0.0,
        "cost":cost,"pairs":pairs_cids,
    }
    with open(output_dir/"summary.json","w") as f:
        json.dump(summary_dict,f,indent=2,default=str)
    metadata={
        "seed":int(cfg.seed),"device":device,"platform":platform.platform(),"python_version":platform.python_version(),
        "torch_version":torch.__version__,"config":dataclasses.asdict(cfg),
        "pairing_mode":pairing_mode,"pairing_tau":pairing_tau,
        "pairs":pairs_cids,"singletons":singletons_cids,
    }
    with open(output_dir/"metadata.json","w") as f:
        json.dump(metadata,f,indent=2,default=str)
    pd.DataFrame(agg_rows).to_csv(output_dir/"pair_aggregate.csv", index=False)  # ensure exists
    print(f"[Dissimilarity] completed seed {cfg.seed} pairing {pairing_mode} tau {pairing_tau} pairs {pairs_cids} meanHV {summary_dict['mean_hv']:.4f}")

    return {"output_dir":output_dir,"pair_results":pair_results,"cost":cost,"summary":summary_dict,"similarity_matrix":S,"distance_matrix":D}
