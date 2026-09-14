"""Matching clients by similarity: maximize intra-pair similarity.

For N clients we want ~N/2 pairs maximizing sum of similarities.
This is classic maximum weight matching on a general graph (Edmonds' blossom).
We use networkx.max_weight_matching for global optimum (O(N^3)), not greedy.

Odd N: one singleton group (always represented by itself). Documented decision.
"""
from typing import List, Tuple
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


def _greedy_matching(S: np.ndarray) -> List[Tuple[int, int]]:
    """Fallback greedy: sort edges by weight descending and pick non-conflicting.
    Better than simple but not guaranteed optimal; used only if networkx unavailable.
    """
    n = S.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((S[i, j], i, j))
    edges.sort(reverse=True)
    used = set()
    pairs = []
    for w, i, j in edges:
        if i not in used and j not in used:
            pairs.append((int(i), int(j)))
            used.add(i)
            used.add(j)
    return pairs


def max_similarity_matching(S: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Find pairing maximizing sum of similarities.

    Returns:
      pairs: list of (a,b) tuples sorted
      singletons: list of unmatched client indices (0..N-1)

    For even N, singletons is empty. For odd N, one element.

    Uses networkx.max_weight_matching with maxcardinality=True to ensure
    maximum number of pairs (floor(N/2)) with maximum total weight.
    Falls back to greedy if networkx not available.
    """
    n = S.shape[0]
    if n == 0:
        return [], []
    if n == 1:
        return [], [0]
    if n == 2:
        return [(0, 1)], []

    if HAS_NX:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        # weights: shift to non-negative to avoid negative-weight skipping edge case.
        # We use S directly but add offset if needed; networkx handles negative with maxcardinality=True
        # but to be safe ensure positivity by adding 1.1 (range becomes 0.1..2.1)
        # Preserve ordering: w' = S + 1.1, still maximizes S.
        offset = 1.1
        for i in range(n):
            for j in range(i + 1, n):
                w = float(S[i, j] + offset)
                G.add_edge(i, j, weight=w)
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")
        # matching is set of (u,v)
        pairs = [tuple(sorted((int(u), int(v)))) for u, v in matching]
        pairs = sorted(pairs)
        # Consistency check: no node appears twice, no self-loops, max cardinality
        used = set()
        for a, b in pairs:
            assert a != b, "self-pair"
            assert a not in used and b not in used, "duplicate node"
            used.update([a, b])
        # singletons: nodes not in used
        singletons = [i for i in range(n) if i not in used]
        # For even N, singletons should be empty
        # For odd N, exactly 1
        return pairs, singletons
    else:
        # fallback greedy (better than nothing, but not optimal)
        pairs = _greedy_matching(S)
        used = set()
        for a, b in pairs:
            used.update([a, b])
        singletons = [i for i in range(n) if i not in used]
        return pairs, singletons


def get_representatives(
    pairs: List[Tuple[int, int]],
    singletons: List[int],
    generation: int,
) -> List[int]:
    """Alternating representatives: per generation pick one per pair.

    Generation even -> first element of each pair.
    Generation odd  -> second element.
    Singleton always itself.

    Deterministic per generation for reproducibility.
    """
    reps = []
    for a, b in pairs:
        rep = a if (generation % 2 == 0) else b
        reps.append(int(rep))
    for s in singletons:
        reps.append(int(s))
    return sorted(reps)  # sorted for deterministic order, but pairing identity preserved externally


def get_representatives_random(
    pairs: List[Tuple[int, int]],
    singletons: List[int],
    generation: int,
    rng: np.random.RandomState,
) -> List[int]:
    """Random representative per pair per generation (seeded by generation)."""
    # Use generation as part of RNG to be deterministic across runs if rng seeded consistently
    # We clone rng with generation seed to avoid altering global rng state too much? Instead use provided rng
    reps = []
    for a, b in pairs:
        rep = a if rng.rand() < 0.5 else b
        reps.append(int(rep))
    for s in singletons:
        reps.append(int(s))
    return sorted(reps)


def random_pairing(S: np.ndarray, rng: np.random.RandomState) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Random pairing: shuffle nodes and pair sequentially. Maximizes randomness, baseline."""
    n = S.shape[0]
    if n <= 1:
        return max_similarity_matching(S)  # fallback
    nodes = list(range(n))
    rng.shuffle(nodes)
    pairs = []
    singletons = []
    i = 0
    while i + 1 < len(nodes):
        a, b = int(nodes[i]), int(nodes[i + 1])
        pairs.append(tuple(sorted((a, b))))
        i += 2
    if i < len(nodes):
        singletons = [int(nodes[i])]
    pairs = sorted(pairs)
    return pairs, singletons


def probabilistic_similarity_pairing(
    S: np.ndarray, tau: float, rng: np.random.RandomState
) -> Tuple[List[Tuple[int, int]], List[int], dict]:
    """Probabilistic pairing favoring similar clients.

    For each step, pick a random unmatched node as anchor, then sample its partner
    from remaining unmatched nodes with probability P(i,j) ∝ exp(sim(i,j)/τ).

    τ small → peaked (near-deterministic max similarity)
    τ large → uniform (more stochastic)

    Returns pairs, singletons, info dict with entropy and mean prob concentration.
    """
    n = S.shape[0]
    if n <= 1:
        return max_similarity_matching(S) + ({},)
    if n == 2:
        return [(0, 1)], [], {"entropy": 0.0, "tau": tau}

    tau = float(tau)
    # Handle extreme tau: avoid overflow
    # Clip sim to [-1,1] already, exp(sim/tau) could overflow for tau very small (≈0.01 => exp(100)=~1e43)
    # We'll normalize via subtract max trick per sampling step.

    unmatched = set(range(n))
    pairs = []
    entropies = []
    max_probs = []

    # To make deterministic given rng, we shuffle order of anchor selection randomly but via rng
    while len(unmatched) >= 2:
        # pick anchor randomly from unmatched (uniform)
        anchor = int(rng.choice(list(unmatched)))
        remaining = [j for j in unmatched if j != anchor]
        # compute logits = sim/τ
        sims = np.array([float(S[anchor, j]) for j in remaining], dtype=float)
        # numerical stability: subtract max before exp
        logits = sims / max(tau, 1e-9)
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        probs = exps / exps.sum()
        # entropy
        ent = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(float(ent))
        max_probs.append(float(probs.max()))
        # sample partner
        partner_idx = int(rng.choice(len(remaining), p=probs))
        partner = int(remaining[partner_idx])
        pairs.append(tuple(sorted((anchor, partner))))
        unmatched.remove(anchor)
        unmatched.remove(partner)

    singletons = sorted(list(unmatched))
    pairs = sorted(pairs)
    info = {
        "tau": float(tau),
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "max_prob_mean": float(np.mean(max_probs)) if max_probs else 0.0,
        "n_steps": len(pairs),
    }
    return pairs, singletons, info


def max_dissimilarity_matching(S: np.ndarray, D: np.ndarray = None) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Find pairing maximizing sum of distances (minimizing similarity).

    Equivalent to max_weight_matching on D = 1 - S.
    If D not provided, computes as 1 - S.
    """
    n = S.shape[0]
    if n == 0:
        return [], []
    if n == 1:
        return [], [0]
    if n == 2:
        return [(0, 1)], []
    if D is None:
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
    if HAS_NX:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        # D in [0,2], add offset 0.1 to keep positive but preserve order
        offset = 0.1
        for i in range(n):
            for j in range(i + 1, n):
                w = float(D[i, j] + offset)
                G.add_edge(i, j, weight=w)
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")
        pairs = [tuple(sorted((int(u), int(v)))) for u, v in matching]
        pairs = sorted(pairs)
        used = set()
        for a, b in pairs:
            assert a != b
            assert a not in used and b not in used
            used.update([a, b])
        singletons = [i for i in range(n) if i not in used]
        return pairs, singletons
    else:
        # fallback greedy on D
        n = D.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((D[i, j], i, j))
        edges.sort(reverse=True)
        used = set()
        pairs = []
        for w, i, j in edges:
            if i not in used and j not in used:
                pairs.append((int(i), int(j)))
                used.add(i); used.add(j)
        singletons = [i for i in range(n) if i not in used]
        return pairs, singletons


def probabilistic_dissimilarity_pairing(
    S: np.ndarray, tau: float, rng: np.random.RandomState, D: np.ndarray = None
) -> Tuple[List[Tuple[int, int]], List[int], dict]:
    """Probabilistic pairing favoring DISSIMILAR clients.

    P(i,j) ∝ exp(dist(i,j)/τ)  where dist = 1 - sim  (or provided D).
    """
    n = S.shape[0]
    if D is None:
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
    if n <= 1:
        return max_dissimilarity_matching(S, D) + ({},)
    if n == 2:
        return [(0, 1)], [], {"entropy": 0.0, "tau": float(tau)}

    tau = float(tau)
    unmatched = set(range(n))
    pairs = []
    entropies = []
    max_probs = []
    while len(unmatched) >= 2:
        anchor = int(rng.choice(list(unmatched)))
        remaining = [j for j in unmatched if j != anchor]
        dists = np.array([float(D[anchor, j]) for j in remaining], dtype=float)
        logits = dists / max(tau, 1e-9)
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        probs = exps / exps.sum()
        ent = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(float(ent))
        max_probs.append(float(probs.max()))
        partner_idx = int(rng.choice(len(remaining), p=probs))
        partner = int(remaining[partner_idx])
        pairs.append(tuple(sorted((anchor, partner))))
        unmatched.remove(anchor); unmatched.remove(partner)
    singletons = sorted(list(unmatched))
    pairs = sorted(pairs)
    info = {
        "tau": float(tau),
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "max_prob_mean": float(np.mean(max_probs)) if max_probs else 0.0,
        "n_steps": len(pairs),
    }
    return pairs, singletons, info


def intra_pair_similarities(S: np.ndarray, pairs: List[Tuple[int, int]]) -> List[float]:
    """Similarity within each pair."""
    return [float(S[a, b]) for a, b in pairs]


def intra_pair_distances(D: np.ndarray, pairs: List[Tuple[int, int]]) -> List[float]:
    return [float(D[a, b]) for a, b in pairs]


def matching_summary(S: np.ndarray, pairs: List[Tuple[int, int]], singletons: List[int], D: np.ndarray = None) -> dict:
    sims = intra_pair_similarities(S, pairs) if pairs else []
    if D is None:
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
    dists = intra_pair_distances(D, pairs) if pairs else []
    return {
        "n_clients": int(S.shape[0]),
        "n_pairs": int(len(pairs)),
        "n_singletons": int(len(singletons)),
        "pairs": [[int(a), int(b)] for a, b in pairs],
        "singletons": [int(s) for s in singletons],
        "intra_pair_similarities": sims,
        "intra_pair_distances": dists,
        "intra_pair_mean": float(np.mean(sims)) if sims else 0.0,
        "intra_pair_min": float(np.min(sims)) if sims else 0.0,
        "intra_pair_max": float(np.max(sims)) if sims else 0.0,
        "intra_pair_std": float(np.std(sims)) if sims else 0.0,
        "intra_pair_distance_mean": float(np.mean(dists)) if dists else 0.0,
        "intra_pair_distance_min": float(np.min(dists)) if dists else 0.0,
        "intra_pair_distance_max": float(np.max(dists)) if dists else 0.0,
    }
