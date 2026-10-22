"""Build references.bib, coded extraction table, counts and figures from search/ outputs."""
import json, re, sys, unicodedata
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "search"); from include import THEMES
d = pd.read_csv("search/screened.csv"); inc = json.load(open("search/included.json"))
ok = {i for t in THEMES.values() for i in t} - {238}  # 238: arXiv v1 title of 231 (same work)
inc = [x for x in inc if x["idx"] in ok]

def ascii_(s): return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
def key(x):
    last = ascii_(x["authors"][0].split()[-1]).lower() if x["authors"] else "anon"
    return re.sub(r"\W", "", last) + str(x["year"]) + re.sub(r"\W", "", ascii_(x["title"]).split()[0].lower())
def esc(s): return ascii_(s or "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
seen, bib = set(), []
for x in inc:
    k = key(x)
    while k in seen: k += "b"
    seen.add(k); x["key"] = k
    au = (" and ".join(ascii_(a).strip() for a in x["authors"][:12]) + (" and others" if len(x["authors"]) > 12 else "")).strip() or "Anonymous"
    t = "article" if x["type"] == "article" else "misc"
    f = [f"  author = {{{au}}}", f"  title = {{{{{esc(x['title'])}}}}}", f"  year = {{{x['year']}}}"]
    if x["venue"] and t == "article":
        f.append(f"  journal = {{{esc(x['venue'])}}}")
        if x["volume"]: f.append(f"  volume = {{{x['volume']}}}")
        if x["pages"]: f.append(f"  pages = {{{x['pages']}}}")
    elif t == "misc": f.append(f"  howpublished = {{{esc(x['venue'] or 'preprint')}}}")
    if x["doi"]: f.append(f"  doi = {{{x['doi']}}}")
    bib.append(f"@{t}{{{k},\n" + ",\n".join(f) + "\n}")
open("references_included.bib", "w").write("\n\n".join(bib) + "\n")

R = lambda p: re.compile(p, re.I)
P = dict(evo=R(r"evolution|genetic|NSGA|MOEA|swarm|\bPSO\b|particle swarm|genetic algorithm"),
         supernet=R(r"supernet|super-net|weight[- ]sharing|one-shot|sub-?net"),
         diff=R(r"differentiable|DARTS|gradient-based search"),
         rl=R(r"reinforcement|\bDQN\b|DDPG|\bPPO\b|bandit|Q-learning|\bD4PG\b"),
         moo=R(r"multi-?objective|many-?objective|pareto"),
         pers=R(r"personali[sz]"), res=R(r"resource|hardware|energy|latency|edge|mobile"),
         noniid=R(r"non-?iid|non-independent|heterogene"),
         fair=R(r"fair|worst-?(client|case|performing)|equitab|consistency"),
         sel=R(r"client selection|participant selection|selecting clients|client sampling|device selection|selection of clients|participant|sampled clients|sampling"),
         sim=R(r"similarity|cluster|divergen|dissimilar|cosine|label distribution|gradient"),
         cost=R(r"communication|computational cost|overhead|payload|efficien"))
rows = []
for x in inc:
    r = d.loc[x["idx"]]; txt = f"{r.title} {r.abstract}"
    theme = [t for t, ids in THEMES.items() if x["idx"] in ids]
    rows.append(dict(key=x["key"], idx=x["idx"], year=x["year"], peer=(x["type"] == "article"), theme=theme[0],
                     nas=bool(re.search(r"architecture search|\bNAS\b", txt, re.I)), **{k: bool(p.search(txt)) for k, p in P.items()}))
c = pd.DataFrame(rows); c.to_csv("search/coded.csv", index=False)
num = {}
num["N"] = len(c); num["peer"] = int(c.peer.sum()); num["pre"] = int((~c.peer).sum())
for t in THEMES: num["th_" + t] = int((c.theme == t).sum())
num["nas"] = int(c.nas.sum()); num["nas_evo"] = int((c.nas & c.evo).sum()); num["nas_super"] = int((c.nas & c.supernet).sum())
num["nas_pers"] = int((c.nas & c.pers).sum()); num["nas_res"] = int((c.nas & c.res).sum()); num["nas_noniid"] = int((c.nas & c.noniid).sum())
num["nas_moo"] = int((c.nas & c.moo).sum()); num["nas_sel"] = int((c.nas & c.sel).sum()); num["nas_fair"] = int((c.nas & c.fair).sum())
num["nas_sim"] = int((c.nas & c.sim & c.sel).sum()); num["nas_evo_sel"] = int((c.nas & c.evo & c.sel).sum())
num["evo_sel"] = int((c.evo & c.sel).sum()); num["moo_sel"] = int((c.moo & c.sel).sum()); num["sel_total"] = int(c.sel.sum())
num["sel_sim"] = int((c.sel & c.sim).sum()); num["sel_fair"] = int((c.sel & c.fair).sum()); num["rl_sel"] = int((c.rl & c.sel).sum())
num["year_min"] = int(c.year.min()); num["recent"] = int((c.year >= 2024).sum())
json.dump(num, open("search/numbers.json", "w"), indent=1); print(num)
with open("numbers.tex", "w") as f:
    for k, v in num.items(): f.write(f"\\newcommand{{\\n{k.replace('_','')}}}{{{v}}}\n")
pr = json.load(open("search/prisma.json"))
for k, v in pr.items(): open("numbers.tex", "a").write(f"\\newcommand{{\\p{k.replace('_','')}}}{{{v}}}\n")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
y = c.groupby(["year", "theme"]).size().unstack(fill_value=0)
y.plot.bar(stacked=True, ax=ax[0], color=["#4C72B0", "#DD8452", "#55A868"]); ax[0].set_xlabel("Year"); ax[0].set_ylabel("Included records"); ax[0].legend(title="Theme", fontsize=7)
labels = ["NAS", "+evolutionary", "+supernet/\nweight-sharing", "+personalized", "+client\nselection", "+fairness/\nworst-client"]
vals = [num["nas"], num["nas_evo"], num["nas_super"], num["nas_pers"], num["nas_sel"], num["nas_fair"]]
ax[1].bar(labels, vals, color="#4C72B0"); ax[1].set_ylabel("Records (of NAS subset)"); ax[1].tick_params(axis="x", labelsize=7)
for i, v in enumerate(vals): ax[1].text(i, v + .3, v, ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("figures/corpus_overview.pdf"); print("fig ok")

# appendix table
yn = lambda b: r"\checkmark" if b else ""
with open("included_table.tex", "w") as f:
    f.write("\\begin{longtable}{p{0.27\\linewidth}rcl cccccc}\n\\caption{Included records with abstract-coded attributes (approximate; see Section~\\ref{sec:methods}). NAS: architecture search; Evo: evolutionary/swarm; MOO: multi-/many-objective; Sel: client/participant selection or sampling; Sim: similarity/clustering/divergence; Fair: fairness/worst-client.}\\label{tab:included}\\\\\n\\toprule\nRecord & Year & Peer & Theme & NAS & Evo & MOO & Sel & Sim & Fair\\\\\n\\midrule\\endfirsthead\n\\toprule Record & Year & Peer & Theme & NAS & Evo & MOO & Sel & Sim & Fair\\\\\\midrule\\endhead\n")
    for _, r in c.sort_values(["theme", "year"]).iterrows():
        th = {"fednas": "FedNAS", "evo_moo": "Evo/MOO", "selection": "Selection"}[r.theme]
        f.write(f"\\cite{{{r.key}}} & {r.year} & {yn(r.peer)} & {th} & {yn(r.nas)} & {yn(r.evo)} & {yn(r.moo)} & {yn(r.sel)} & {yn(r.sim)} & {yn(r.fair)}\\\\\n")
    f.write("\\bottomrule\n\\end{longtable}\n")
