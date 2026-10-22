import json,numpy as np,pandas as pd,torch,sys
from sklearn.metrics import adjusted_rand_score as ari
from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta
from evofederated.models.cnn_genome import CNNGenome, cnn_to_model
from evofederated.utils.config import load_config
rows=[]
for seed in range(5):
    cfg=load_config('configs/cs_e3.yaml'); cfg.dataset.seed=seed; cfg.seed=seed; cfg.probe.seed=seed
    torch.set_num_threads(2)
    clients,info=build_federated_clients(cfg,seed=seed); shape=tuple(clients[0].train_dataset[0][0].shape); K=info['n_classes']
    probe=lambda: cnn_to_model(CNNGenome(2,[8,16],3,'relu',0.0,True),shape,K)
    saved=np.array(json.load(open(f'runs/e3/cs_nsga/seed{seed}/clusters.json'))['labels'])
    d1=probe_deltas(clients,probe,1,cfg.train.lr,cfg.train.optimizer,seed)
    l1=cluster_delta(d1,10,'kmeans',seed)
    d2=probe_deltas(clients,probe,1,cfg.train.lr,cfg.train.optimizer,seed+100)  # otra inicialización/orden de la sonda
    l2=cluster_delta(d2,10,'kmeans',seed)
    l3=cluster_delta(d1,10,'kmeans',seed+7)  # otra semilla de k-means, mismos Δ
    rows.append({'seed':seed,'ARI_rebuild_vs_saved(2 threads)':ari(saved,l1),'ARI_otra_sonda':ari(l1,l2),'ARI_otro_kmeans_seed':ari(l1,l3)})
    print(rows[-1],flush=True)
print(pd.DataFrame(rows).mean().round(3).to_dict())
