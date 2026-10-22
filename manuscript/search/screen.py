"""Deduplicate + rule-based screening of raw OpenAlex hits. Writes search/screened.csv and search/prisma.json."""
import json, re, pandas as pd

d = json.load(open("search/raw_openalex.json"))
W = d["works"]; n_raw = sum(len(set(w["queries"])) for w in W)

def abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv: return ""
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    return " ".join(t for _, t in pos)

TOPICS = {
    "nas": r"architecture search|\bNAS\b|neural architecture|architecture design|model architecture",
    "evo_moo": r"evolutionary|genetic algorithm|NSGA|MOEA|multi-?objective|many-?objective|pareto|swarm",
    "selection": r"client selection|participant selection|selecting clients|client sampling|device selection|node selection",
    "similarity": r"similarity|cluster|divergen|dissimilar|gradient|cosine",
    "fairness": r"fairness|worst-?client|worst-?case|minimax|robust|personali[sz]",
    "cost": r"communication cost|computational cost|efficien|overhead|reduce.*cost|resource",
}
rows = []
for w in W:
    ab = abstract(w); txt = f"{w['title'] or ''} {ab}"
    loc = (w.get("primary_location") or {}).get("source") or {}
    rows.append(dict(id=w["id"], doi=(w.get("doi") or "").replace("https://doi.org/", ""), title=w["title"],
        year=w["publication_year"], type=w["type"], venue=loc.get("display_name"), cites=w["cited_by_count"],
        first_author=(w["authorships"][0]["author"]["display_name"] if w["authorships"] else ""),
        n_queries=len(set(w["queries"])), queries="|".join(sorted(set(w["queries"]))), has_abstract=bool(ab),
        federated=bool(re.search(r"federated", txt, re.I)),
        **{t: bool(re.search(p, txt, re.I)) for t, p in TOPICS.items()}, abstract=ab))
df = pd.DataFrame(rows)
n_unique = len(df)
df["title_key"] = df.title.fillna("").str.lower().str.replace(r"\W+", "", regex=True)
df = df.sort_values("cites", ascending=False).drop_duplicates("title_key")
n_dedup = len(df)
s1 = df[df.federated & df.has_abstract]
# topical score: primary = (nas | evo_moo | selection | similarity | fairness) ; require >= 2 topic flags or NAS
topic_cols = list(TOPICS)
s1 = s1.assign(score=s1[topic_cols].sum(axis=1))
s2 = s1[(s1.nas & (s1.evo_moo | s1.cost | s1.selection)) | (s1.score >= 3) | (s1.selection & s1.similarity) | (s1.evo_moo & s1.selection)]
s2 = s2.sort_values(["score", "cites"], ascending=False)
pr = dict(records_identified=n_raw, unique_openalex_ids=n_unique, after_title_dedup=n_dedup,
          after_federated_abstract_filter=len(s1), after_topical_filter=len(s2))
json.dump(pr, open("search/prisma.json", "w"), indent=1); print(pr)
s2.drop(columns="title_key").to_csv("search/screened.csv", index=False)
