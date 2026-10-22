"""Print keyword-matched sentences per checklist category for one full-text file (reading aid, not a classifier)."""
import re, sys
CATS = {
 "repeat/seeds": r"\bseeds?\b|independent runs|\b\d+ runs\b|repeated|trials|standard deviation|std\b|confidence interval|mean (and|±)|\bt-test|wilcoxon|averaged over",
 "clients/partition": r"dirichlet|\balpha\b|α|non-?iid|(\d+|ten|twenty|hundred) (clients|devices|workers|parties|users)|clients? (per|in each) round|partition",
 "val/test split": r"validation (set|data|split|accuracy)|held-?out|test (set|data|accuracy)|hold.?out|validation and test|train(ing)?/val",
 "participation": r"sampl(e|ed|ing) (of )?(clients|devices|participants|a subset)|subset of clients|participat|randomly (select|sampl)|selected clients|client selection|fraction",
 "search budget/cost": r"gpu[- ]?(hours|days)|search cost|search time|number of (evaluations|architectures|candidates|rounds)|population|generations?|wall[- ]clock|flops|communication (cost|overhead|rounds)|rounds",
 "baselines": r"random search|baseline|compared (with|to)|state-of-the-art|fedavg|fednas|darts",
 "objective": r"worst|fairness|minimum accuracy|per-client|each client|pareto|hypervolume|multi-?objective|fitness",
}
txt = re.sub(r"\s+", " ", open(sys.argv[1]).read())
sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", txt)
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 5
only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CATS)
for c, p in [(c, CATS[c]) for c in only]:
    rx = re.compile(p, re.I); hits = [s for s in sents if rx.search(s) and 40 < len(s) < 420]
    print(f"\n## {c} ({len(hits)} hits)")
    step = max(1, len(hits) // lim)
    for s in hits[::step][:lim]: print(" -", s)
