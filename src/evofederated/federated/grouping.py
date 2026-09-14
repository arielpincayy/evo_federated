"""Grouping clients for objective reduction.

Provides functions to create G groups from N clients using similarity/dissimilarity/random.

For N=8:
  per_silo: 8 objectives
  grouped N/2: 4 groups size 2
  grouped N/4: 2 groups size 4

For odd N, groups may be slightly unbalanced but we keep size approx.

Strategies:
- similarity: maximize intra-group similarity (greedy matching + merging)
- dissimilarity: maximize intra-group distance (opposite)
- random: random shuffle then chunk

Aggregation per group:
- mean: fitness = mean accuracy of members
- min: fitness = min accuracy (fairness focused)
"""
from typing import List, Dict, Tuple
import numpy as np

def create_groups(S: np.ndarray, n_groups: int, strategy: str = "similarity", rng: np.random.RandomState = None, D: np.ndarray = None) -> List[List[int]]:
    """
    Returns list of groups, each group is list of client pos indices (0..N-1).
    n_groups must divide N or close: we will create groups of approx size N//n_groups.
    
    For n_groups == N -> per_silo (each group size 1)
    For n_groups == N//2 -> pairs
    For n_groups == N//4 -> quads

    Strategy similarity: try to group similar clients together.
    """
    N = S.shape[0]
    if D is None:
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
    if n_groups >= N:
        return [[i] for i in range(N)]
    if n_groups <= 0:
        return [list(range(N))]
    
    group_size = N // n_groups
    remainder = N % n_groups
    # For simplicity, we will handle exact division cases only for benchmark (N=4,8,16,32 and n_groups N, N/2, N/4 always divide exactly for powers of 2)
    # For remainder, distribute 1 extra to first groups
    if remainder != 0 and strategy != "random":
        # fallback to random for remainder case
        pass

    if strategy == "random":
        if rng is None:
            rng = np.random.RandomState(0)
        perm = list(range(N))
        rng.shuffle(perm)
        groups=[]
        idx=0
        for g in range(n_groups):
            sz = group_size + (1 if g < remainder else 0)
            groups.append(sorted(perm[idx:idx+sz]))
            idx+=sz
        return groups

    elif strategy == "similarity":
        # Greedy: start with most similar pair, then expand.
        # For size 2, use max weight matching.
        if group_size == 2 and N % 2 == 0:
            from .matching import max_similarity_matching
            pairs, _ = max_similarity_matching(S)
            return [list(p) for p in pairs]
        elif group_size == 4 and N % 4 == 0:
            # Two-stage: first make pairs via similarity, then pair pairs by mean inter-pair similarity
            from .matching import max_similarity_matching
            pairs, _ = max_similarity_matching(S)
            # pairs is list of 2-tuples, number = N/2
            # Compute similarity between pairs as mean of cross similarities
            n_pairs = len(pairs)
            # Build pair-pair similarity matrix
            pair_S = np.zeros((n_pairs, n_pairs))
            for i in range(n_pairs):
                for j in range(n_pairs):
                    if i==j: continue
                    a,b = pairs[i]; c,d = pairs[j]
                    vals = [S[a,c], S[a,d], S[b,c], S[b,d]]
                    pair_S[i,j] = np.mean(vals)
            # Now match pair-pairs to form quads: max similarity matching on pair_S
            try:
                import networkx as nx
                G = nx.Graph()
                G.add_nodes_from(range(n_pairs))
                for i in range(n_pairs):
                    for j in range(i+1, n_pairs):
                        w = float(pair_S[i,j] + 1.1)
                        G.add_edge(i,j, weight=w)
                matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")
                quads=[]
                used=set()
                for u,v in matching:
                    u=int(u); v=int(v)
                    quad = sorted(list(pairs[u]) + list(pairs[v]))
                    quads.append(quad)
                    used.add(u); used.add(v)
                # any leftover (odd number of pairs shouldn't happen for N divisible by 4)
                for i in range(n_pairs):
                    if i not in used:
                        quads.append(list(pairs[i]))
                return quads
            except Exception:
                # fallback greedy
                remaining = list(range(n_pairs))
                quads=[]
                while len(remaining)>=2:
                    # pick first, find most similar remaining
                    i = remaining[0]
                    best_j = max(remaining[1:], key=lambda j: pair_S[i,j])
                    quads.append(sorted(list(pairs[i]) + list(pairs[best_j])))
                    remaining.remove(i); remaining.remove(best_j)
                if remaining:
                    # leftover single pair -> add as separate group (size2) but we want quads, so distribute?
                    quads.append(list(pairs[remaining[0]]))
                return quads
        else:
            # Generic hierarchical clustering greedy for arbitrary sizes
            # Use simple approach: sorted edges by similarity descending, then union-find to build groups until size limit
            # Initialize each node as its own group, then merge most similar groups iteratively (single linkage mean)
            groups=[[i] for i in range(N)]
            # compute group-group similarity as mean of cross S
            # iteratively merge most similar pair of groups while we have more groups than n_groups
            while len(groups) > n_groups:
                best_i=-1; best_j=-1; best_sim=-2
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        # don't merge if would exceed max size? allow slight overflow but keep balanced
                        if len(groups[i])+len(groups[j]) > group_size + (1 if remainder>0 else 0):
                            # if strict, skip but if no other option, allow
                            # check if any feasible pair exists
                            continue
                        # mean similarity between groups
                        sims=[]
                        for a in groups[i]:
                            for b in groups[j]:
                                sims.append(S[a,b])
                        mean_sim=np.mean(sims) if sims else -1
                        if mean_sim > best_sim:
                            best_sim=mean_sim; best_i=i; best_j=j
                if best_i==-1:
                    # no feasible merge without exceeding size, allow any merge with smallest overflow
                    best_i, best_j = 0,1
                    best_sim = -1
                    for i in range(len(groups)):
                        for j in range(i+1, len(groups)):
                            sims=[S[a,b] for a in groups[i] for b in groups[j]]
                            mean_sim=np.mean(sims)
                            if mean_sim > best_sim:
                                best_sim=mean_sim; best_i=i; best_j=j
                # merge
                new_group = sorted(groups[best_i] + groups[best_j])
                # remove higher index first
                if best_i > best_j:
                    groups.pop(best_i); groups.pop(best_j)
                else:
                    groups.pop(best_j); groups.pop(best_i)
                groups.append(new_group)
            return sorted(groups)

    elif strategy == "dissimilarity":
        # Similar but maximize distance
        if group_size == 2 and N % 2 == 0:
            from .matching import max_dissimilarity_matching
            pairs,_ = max_dissimilarity_matching(S, D)
            return [list(p) for p in pairs]
        elif group_size == 4 and N % 4 == 0:
            from .matching import max_dissimilarity_matching
            pairs,_ = max_dissimilarity_matching(S, D)
            n_pairs=len(pairs)
            pair_D=np.zeros((n_pairs,n_pairs))
            for i in range(n_pairs):
                for j in range(n_pairs):
                    if i==j: continue
                    a,b=pairs[i]; c,d=pairs[j]
                    vals=[D[a,c], D[a,d], D[b,c], D[b,d]]
                    pair_D[i,j]=np.mean(vals)
            # match most dissimilar pair-pairs (max distance)
            try:
                import networkx as nx
                G=nx.Graph()
                G.add_nodes_from(range(n_pairs))
                for i in range(n_pairs):
                    for j in range(i+1,n_pairs):
                        w=float(pair_D[i,j]+0.1)
                        G.add_edge(i,j,weight=w)
                matching=nx.algorithms.matching.max_weight_matching(G,maxcardinality=True,weight="weight")
                quads=[]
                used=set()
                for u,v in matching:
                    u=int(u);v=int(v)
                    quads.append(sorted(list(pairs[u])+list(pairs[v])))
                    used.add(u); used.add(v)
                for i in range(n_pairs):
                    if i not in used:
                        quads.append(list(pairs[i]))
                return quads
            except Exception:
                remaining=list(range(n_pairs))
                quads=[]
                while len(remaining)>=2:
                    i=remaining[0]
                    best_j=max(remaining[1:], key=lambda j: pair_D[i,j])
                    quads.append(sorted(list(pairs[i])+list(pairs[best_j])))
                    remaining.remove(i); remaining.remove(best_j)
                if remaining:
                    quads.append(list(pairs[remaining[0]]))
                return quads
        else:
            groups=[[i] for i in range(N)]
            while len(groups) > n_groups:
                best_i=-1; best_j=-1; best_dist=-1
                for i in range(len(groups)):
                    for j in range(i+1,len(groups)):
                        if len(groups[i])+len(groups[j]) > group_size + (1 if remainder>0 else 0):
                            continue
                        dists=[D[a,b] for a in groups[i] for b in groups[j]]
                        mean_dist=np.mean(dists) if dists else -1
                        if mean_dist > best_dist:
                            best_dist=mean_dist; best_i=i; best_j=j
                if best_i==-1:
                    best_i,best_j=0,1
                    best_dist=-1
                    for i in range(len(groups)):
                        for j in range(i+1,len(groups)):
                            dists=[D[a,b] for a in groups[i] for b in groups[j]]
                            mean_dist=np.mean(dists)
                            if mean_dist>best_dist:
                                best_dist=mean_dist; best_i=i; best_j=j
                new_group=sorted(groups[best_i]+groups[best_j])
                if best_i>best_j:
                    groups.pop(best_i); groups.pop(best_j)
                else:
                    groups.pop(best_j); groups.pop(best_i)
                groups.append(new_group)
            return sorted(groups)
    else:
        raise ValueError(f"Unknown grouping strategy {strategy}")

def grouping_summary(groups: List[List[int]], S: np.ndarray, D: np.ndarray = None) -> dict:
    if D is None:
        D=1.0 - S
        np.fill_diagonal(D,0.0)
    intra_sims=[]
    intra_dists=[]
    for g in groups:
        if len(g)<2:
            continue
        # mean intra similarity
        vals=[]
        dists=[]
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                vals.append(S[g[i],g[j]])
                dists.append(D[g[i],g[j]])
        if vals:
            intra_sims.append(float(np.mean(vals)))
            intra_dists.append(float(np.mean(dists)))
    return {
        "n_groups": len(groups),
        "groups": groups,
        "group_sizes": [len(g) for g in groups],
        "intra_group_sim_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "intra_group_dist_mean": float(np.mean(intra_dists)) if intra_dists else 0.0,
    }
