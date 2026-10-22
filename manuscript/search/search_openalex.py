"""Run the review's search strings against OpenAlex and save raw hits (JSON) + a log."""
import json, time, datetime, requests

YEARS = "2019-01-01"
QUERIES = {
    "Q1_fed_nas": '"federated neural architecture search"',
    "Q2_evo_fed_nas": '(federated AND (evolutionary OR "genetic algorithm" OR NSGA) AND "architecture search")',
    "Q3_moo_fl": '(federated learning AND ("multi-objective" OR "multiobjective" OR "many-objective") AND (evolutionary OR Pareto))',
    "Q4_client_sel": '("client selection" OR "participant selection") AND "federated learning" AND (heterogeneous OR "non-IID")',
    "Q5_similarity": '("client clustering" OR "gradient similarity" OR "update similarity" OR "cosine similarity") AND "federated learning" AND (selection OR clustering)',
    "Q6_eval_cost": '("neural architecture search" OR "hyperparameter") AND federated AND (cost OR efficient OR "partial evaluation" OR "low-fidelity")',
    "Q7_worst_client": '("federated learning") AND (fairness OR "worst-client" OR "worst-case" OR "minimax") AND (noniid OR "non-IID" OR heterogeneity)',
}
rows, log = {}, []
for qid, q in QUERIES.items():
    n = 0
    for page in (1, 2):  # top-200 by relevance per query
        r = requests.get("https://api.openalex.org/works", params={
            "search": q, "per-page": 100, "page": page, "sort": "relevance_score:desc",
            "filter": f"from_publication_date:{YEARS},type:article|preprint|book-chapter|proceedings-article|review",
            "select": "id,doi,title,publication_year,type,primary_location,authorships,cited_by_count,abstract_inverted_index"},
            timeout=60)
        if r.status_code != 200:
            log.append((qid, page, r.status_code, r.text[:200])); break
        res = r.json()
        total = res["meta"]["count"]
        for w in res["results"]:
            rows.setdefault(w["id"], {**w, "queries": []})["queries"].append(qid); n += 1
        time.sleep(1)
    log.append((qid, "hits_total", total if n else 0, "retrieved", n))
json.dump({"date": datetime.date.today().isoformat(), "queries": QUERIES, "log": log, "works": list(rows.values())},
          open("search/raw_openalex.json", "w"))
print(len(rows), log)
