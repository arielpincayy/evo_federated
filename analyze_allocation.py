import numpy as np,pandas as pd,torch
from evofederated.data.federated_split import build_federated_clients
from evofederated.federated.characterization import probe_deltas
from evofederated.federated.clustering import cluster_delta, cluster_labels_oracle
from evofederated.models.cnn_genome import CNNGenome, cnn_to_model
from evofederated.utils.config import load_config
torch.set_num_threads(2)
def alloc(sizes,s,B,mode):
    K=len(sizes)
    if mode=='one': return np.ones(K,int)
    w=sizes*s if mode=='neyman' else sizes.astype(float)
    n=np.maximum(1,np.floor(B*w/w.sum())).astype(int)
    while n.sum()>B: n[np.argmax(n)]-=1
    while n.sum()<B:
        gain=(sizes-n>0)*(w/np.maximum(n,1))
        n[np.argmax(gain)]+=1
    return np.minimum(n,sizes)
def var(g,lab,n):
    N=len(g);tot=0.0
    for c,nc in zip(np.unique(lab),n):
        m=g[lab==c]; sz=len(m); s2=((m-m.mean(0))**2).sum()/max(sz-1,1)
        tot+=(sz/N)**2*(1-nc/sz)/nc*s2
    return tot
def vr(g,B): N=len(g); return (1-B/N)/B*((g-g.mean(0))**2).sum()/(N-1)
rows=[]
for alpha in (0.1,0.5):
  for seed in range(5):
    cfg=load_config('configs/cs_e3.yaml'); cfg.dataset.alpha=alpha; cfg.dataset.seed=seed; cfg.seed=seed; cfg.probe.seed=seed
    clients,info=build_federated_clients(cfg,seed=seed); shape=tuple(clients[0].train_dataset[0][0].shape); K=info['n_classes']
    cd=info['train_partition'].class_distribution.astype(float); g=cd/cd.sum(1,keepdims=True)
    probe=lambda: cnn_to_model(CNNGenome(2,[8,16],3,'relu',0.0,True),shape,K)
    delta=probe_deltas(clients,probe,1,cfg.train.lr,cfg.train.optimizer,seed)
    for B in (10,20):
        base=vr(g,B)
        for src,fn in (('delta',lambda k: cluster_delta(delta,k,'kmeans',seed)),('label_oracle',lambda k: cluster_labels_oracle(cd,k,seed))):
            for Kc in sorted({max(2,B//4),B//2,B}):
                lab=fn(Kc); sizes=np.bincount(lab); s=np.array([np.sqrt(((g[lab==c]-g[lab==c].mean(0))**2).sum()/max(sizes[c]-1,1)) for c in range(Kc)])
                for mode in ('one','proportional','neyman'):
                    if mode=='one' and Kc!=B: continue
                    n=alloc(sizes,s,B,mode); rows.append({'alpha':alpha,'seed':seed,'B':B,'source':src,'K_clusters':Kc,'asignacion':mode,'var_ratio':var(g,lab,n)/base})
    print(alpha,seed,'ok',flush=True)
d=pd.DataFrame(rows); d.to_csv('runs/clusters/allocation_proxy.csv',index=False)
print(d.groupby(['alpha','B','source','K_clusters','asignacion']).var_ratio.mean().round(3).unstack('asignacion').to_string())
