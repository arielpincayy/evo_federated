"""Fetch full author lists for included works (OpenAlex) and verify DOIs against Crossref. Writes search/included.json."""
import json, sys, time, requests
import pandas as pd
sys.path.insert(0, "search"); from include import THEMES
d = pd.read_csv("search/screened.csv")
idx = sorted({i for t in THEMES.values() for i in t})
oa_ids = [d.loc[i, "id"].split("/")[-1] for i in idx]
meta = {}
for k in range(0, len(oa_ids), 40):
    r = requests.get("https://api.openalex.org/works", params={
        "filter": "openalex:" + "|".join(oa_ids[k:k+40]), "per-page": 50,
        "select": "id,doi,title,publication_year,type,primary_location,authorships,biblio"}, timeout=60)
    for w in r.json()["results"]: meta[w["id"]] = w
    time.sleep(1)
out = []
for i in idx:
    w = meta[d.loc[i, "id"]]
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    cr = None
    if doi and not doi.startswith("10.48550") and not doi.startswith("10.5281") and not doi.startswith("10.20944"):
        try:
            c = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30)
            cr = c.status_code == 200 and c.json()["message"]
        except Exception: cr = None
    src = (w.get("primary_location") or {}).get("source") or {}
    out.append(dict(idx=i, doi=doi, title=w["title"], year=w["publication_year"], type=w["type"],
        venue=src.get("display_name"), authors=[a["author"]["display_name"] for a in w["authorships"]],
        volume=w["biblio"].get("volume"), issue=w["biblio"].get("issue"), pages=(w["biblio"].get("first_page") or "") + ("--" + w["biblio"]["last_page"] if w["biblio"].get("last_page") else ""),
        crossref_ok=bool(cr), crossref_title=(cr["title"][0] if cr and cr.get("title") else None)))
    time.sleep(0.1)
json.dump(out, open("search/included.json", "w"), indent=1)
print(len(out), "crossref ok:", sum(o["crossref_ok"] for o in out), "no-crossref:", sum(not o["crossref_ok"] for o in out))
