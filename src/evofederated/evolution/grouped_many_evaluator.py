"""Grouped many-objective evaluator.

Wraps per-siloso full evaluation but aggregates per group.
Supports mean and min aggregation for fairness.
"""
import hashlib, json
import numpy as np
import torch
from pymoo.core.problem import Problem

from ..models.genome import Genome, genome_to_model, model_stats
from ..federated.characterization import flatten_params, set_model_params

def _seed_for_genome(genome):
    h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
    return int(h[:8],16)%(2**31-1)

class GroupedManyObjectiveEvaluator:
    """
    Evaluator with grouped objectives.

    groups: List[List[int]] pos indices (0..N-1) each group.
    aggregation: "mean" or "min"
    objectives: G = len(groups) ; each f_g = 1 - agg_acc(group)
    """
    def __init__(self, clients: dict, input_dim: int, n_classes: int,
                 genome_config: dict, train_config: dict,
                 groups: list, aggregation: str="mean",
                 device="cpu", seed=42):
        self.clients=clients
        self.client_ids_sorted=sorted(clients.keys())
        self.pos_to_cid={pos:cid for pos,cid in enumerate(self.client_ids_sorted)}
        self.groups=groups  # list of list of pos
        self.aggregation=aggregation
        self.n_obj=len(groups)
        self.n_clients=len(clients)
        self.input_dim=input_dim
        self.n_classes=n_classes
        self.genome_config=genome_config
        self.train_config=train_config
        self.device=device
        self.seed=seed
        self.generation=0
        self.rng=np.random.RandomState(seed)
        self.eval_count=0
        self.eval_time_total=0.0
        self.generation_history=[]

    def set_generation(self, gen:int):
        self.generation=int(gen)

    def _train_and_eval(self, genome, client_id, flat_init):
        model=genome_to_model(genome,self.input_dim,self.n_classes)
        set_model_params(model, flat_init)
        client=self.clients[client_id]
        train_res=client.train(model, epochs=self.train_config.get("epochs",5), lr=self.train_config.get("lr",1e-3), optimizer_name=self.train_config.get("optimizer","adam"), device=self.device)
        # Búsqueda en validation; test reservado a evaluación final.
        eval_res=client.evaluate(model, device=self.device, split="val")
        return eval_res, train_res

    def evaluate_single(self, vec, individual_idx, arch_id=None):
        genome=Genome.from_vector(vec,self.genome_config)
        stats=model_stats(genome_to_model(genome,self.input_dim,self.n_classes))
        seed=_seed_for_genome(genome)
        torch.manual_seed(seed)
        template=genome_to_model(genome,self.input_dim,self.n_classes)
        flat_init=flatten_params(template)
        # full evaluation on all clients
        accuracies=[]
        f1s=[]
        losses=[]
        per_client={}
        total_time=0.0
        for pos,cid in enumerate(self.client_ids_sorted):
            eval_res,train_res=self._train_and_eval(genome,cid,flat_init)
            acc=float(eval_res["accuracy"]); f1=float(eval_res["macro_f1"]); loss=float(eval_res["loss"])
            accuracies.append(acc); f1s.append(f1); losses.append(loss)
            per_client[int(cid)]={"accuracy":acc,"f1":f1,"loss":loss,"train_time":float(train_res["time"])}
            self.eval_count+=1
            self.eval_time_total+=float(train_res["time"])
            total_time+=float(train_res["time"])
        # aggregate per group
        group_accs=[]
        group_objs=[]
        for g in self.groups:
            # g is list of pos
            vals=[accuracies[pos] for pos in g]
            if self.aggregation=="mean":
                agg=float(np.mean(vals)) if vals else 0.0
            elif self.aggregation=="min":
                agg=float(np.min(vals)) if vals else 0.0
            else:
                agg=float(np.mean(vals))
            group_accs.append(agg)
            group_objs.append(1.0 - agg)
        F=np.array(group_objs,dtype=float)
        mean_acc=float(np.mean(accuracies))
        median_acc=float(np.median(accuracies))
        worst_acc=float(np.min(accuracies))
        best_acc=float(np.max(accuracies))
        std_acc=float(np.std(accuracies))
        gap=float(best_acc-worst_acc)
        info={
            "genome":genome,"genome_dict":genome.to_dict(),
            "genome_hash":hashlib.md5(json.dumps(genome.to_dict(),sort_keys=True).encode()).hexdigest()[:12],
            "accuracies":accuracies,"group_accuracies":group_accs,"F":F,
            "per_client":per_client,"mean_accuracy":mean_acc,"median_accuracy":median_acc,"worst_accuracy":worst_acc,"best_accuracy":best_acc,"std_accuracy":std_acc,"gap":gap,
            "stats":stats,"elapsed":float(total_time),"generation":int(self.generation),
        }
        rec={
            "generation":int(self.generation),"arch_id":arch_id,"individual_idx":int(individual_idx),
            "genome":json.dumps(genome.to_dict()),"genome_hash":info["genome_hash"],
            "genome_L":genome.L,"genome_hidden":str(genome.hidden_sizes),"genome_act":genome.activation,"genome_dropout":genome.dropout,"genome_bn":genome.use_bn,
            "n_params":stats["n_params"],
            "mean_accuracy":mean_acc,"median_accuracy":median_acc,"worst_accuracy":worst_acc,"best_accuracy":best_acc,"std_accuracy":std_acc,"gap":gap,
            "group_accs":str(group_accs),"groups":str(self.groups),
        }
        for idx,pos_list in enumerate(self.groups):
            rec[f"group_{idx}_acc"]=float(group_accs[idx])
            rec[f"obj_{idx}"]=float(F[idx])
        for idx,cid in enumerate(self.client_ids_sorted):
            rec[f"acc_client_{cid}"]=float(accuracies[idx])
            rec[f"f1_client_{cid}"]=float(f1s[idx])
            rec[f"loss_client_{cid}"]=float(losses[idx])
        self.generation_history.append(rec)
        return F, info

    def evaluate_batch(self, X):
        n=X.shape[0]
        F=np.zeros((n,self.n_obj))
        for i in range(n):
            vec=X[i]
            arch_id=f"g{self.generation}_i{i}"
            f,_=self.evaluate_single(vec,individual_idx=i,arch_id=arch_id)
            F[i]=f
        return F

class GroupedManyProblem(Problem):
    def __init__(self, n_var:int, evaluator:GroupedManyObjectiveEvaluator):
        super().__init__(n_var=n_var, n_obj=evaluator.n_obj, n_constr=0, xl=0.0, xu=1.0)
        self.evaluator=evaluator
    def _evaluate(self, X, out, *args, **kwargs):
        F=self.evaluator.evaluate_batch(X)
        out["F"]=F
