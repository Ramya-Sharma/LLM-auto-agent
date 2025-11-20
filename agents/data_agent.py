# # app/agents/data_agent.py
# import os
# import math
# import time
# import json
# import logging
# from pathlib import Path
# from typing import List, Dict, Optional

# import pandas as pd
# import requests
# from bs4 import BeautifulSoup

# # huggingface
# from huggingface_hub import list_datasets as hf_list_datasets
# from datasets import load_dataset, DatasetDict

# # kaggle
# try:
#     from kaggle.api.kaggle_api_extended import KaggleApi
#     _have_kaggle = True
# except Exception:
#     _have_kaggle = False

# from app.utils.llm_clients import llm_generate_json
# from .synthetic_agent import synthesize_dataset
# from app.storage import ensure_dirs
# from app.utils.run_logger import agent_log

# DATA_DIR = "data"
# ensure_dirs()
# os.makedirs(DATA_DIR, exist_ok=True)

# log = logging.getLogger("data_agent")
# log.setLevel(logging.INFO)

# # ---------------------
# # Helpers / scoring
# # ---------------------
# def _score_candidate(ds_meta: dict, ps: dict) -> float:
#     est_rows = ds_meta.get("est_rows") or 0
#     has_target = 0
#     schema = ds_meta.get("schema") or {}
#     features = schema.get("features") or []
#     target_name = ps.get("target")
#     if target_name and target_name in features:
#         has_target = 1
#     license_ok = 1 if str(ds_meta.get("license","")).lower().startswith(("cc","mit","apache")) else 0
#     score = math.tanh(est_rows/20000.0) * 0.6 + 0.3*has_target + 0.1*license_ok
#     return float(score)

# def _prompt_build_queries(ps: dict) -> str:
#     return f"""
# You are a dataset-finding assistant. Produce a JSON object with a list 'queries' containing 3 short search queries to find public tabular datasets relevant to this problem statement:
# Problem statement: {ps.get('raw_text') or ps}
# Keywords: {ps.get('keywords') or []}
# Return JSON only: {{ "queries": ["...","...","..."] }}
# """

# # ---------------------
# # Hugging Face search
# # ---------------------
# def find_on_hf(run_id: str, query_list: List[str], max_rows:int=5000) -> List[dict]:
#     """
#     Search Hugging Face datasets for names that match tokens in the queries.
#     Returns metadata dicts with local CSV download path where possible.
#     """
#     results = []
#     try:
#         ds_list = hf_list_datasets()  # returns DatasetInfo objects
#     except Exception as e:
#         agent_log(run_id, f"[data_agent] HF list_datasets failed: {e}")
#         ds_list = []

#     # build candidate names by simple token match on the dataset id or title
#     candidates = []
#     for q in query_list:
#         if not q or not str(q).strip():
#             continue
#         q_tokens = str(q).lower().split()
#         for entry in ds_list:
#             try:
#                 # DatasetInfo has .id or .repo_id or string form
#                 name_str = None
#                 if hasattr(entry, "id"):
#                     name_str = entry.id
#                 elif hasattr(entry, "repo_id"):
#                     name_str = entry.repo_id
#                 else:
#                     name_str = str(entry)
#                 if not name_str:
#                     continue
#             except Exception:
#                 name_str = str(entry)
#             name_l = name_str.lower()
#             if all(tok in name_l for tok in q_tokens):
#                 candidates.append(name_str)
#         if len(candidates) >= 8:
#             break

#     # dedupe and limit
#     candidates = list(dict.fromkeys(candidates))[:8]

#     for cand in candidates:
#         try:
#             agent_log(run_id, f"[data_agent] attempting HF load: {cand}")
#             # try load_dataset(cand)
#             ds = load_dataset(cand, split=None)  # may return DatasetDict
#             # choose a sensible split
#             if isinstance(ds, DatasetDict):
#                 split = "train" if "train" in ds.keys() else list(ds.keys())[0]
#                 sub = ds[split]
#             else:
#                 sub = ds
#             df = sub.to_pandas()
#             if len(df) > max_rows:
#                 df = df.sample(max_rows, random_state=0)
#             safe_name = cand.replace("/", "_").replace(":", "_")
#             path = f"{DATA_DIR}/{run_id}_hf_{safe_name}.csv"
#             df.to_csv(path, index=False)
#             meta = {"name": cand, "uri": cand, "license": "unknown", "est_rows": len(df),
#                     "schema": {"features": list(df.columns), "target": None}, "downloaded_to": path}
#             results.append(meta)
#             agent_log(run_id, f"[data_agent] HF candidate saved: {path} rows={len(df)}")
#         except Exception as e:
#             agent_log(run_id, f"[data_agent] HF candidate {cand} skipped: {e}")
#             continue
#     return results

# # ---------------------
# # Kaggle search (optional)
# # ---------------------
# def find_on_kaggle(run_id: str, queries: List[str], max_rows:int=5000) -> List[dict]:
#     results = []
#     if not _have_kaggle:
#         agent_log(run_id, "[data_agent] Kaggle client not available")
#         return results
#     try:
#         api = KaggleApi()
#         api.authenticate()
#     except Exception as e:
#         agent_log(run_id, f"[data_agent] Kaggle auth failed: {e}")
#         return results

#     for q in queries:
#         try:
#             search_res = api.datasets_list(search=q, page=1, max_results=10)
#             for ds in search_res:
#                 try:
#                     slug = f"{ds.ref}"
#                     agent_log(run_id, f"[data_agent] downloading kaggle dataset {slug}")
#                     dest_dir = f"{DATA_DIR}/kaggle_{run_id}"
#                     os.makedirs(dest_dir, exist_ok=True)
#                     api.dataset_download_files(slug, path=dest_dir, unzip=True, quiet=True)
#                     # attempt to find a csv file
#                     csvs = [str(p) for p in Path(dest_dir).glob("**/*.csv")]
#                     if not csvs:
#                         continue
#                     df = pd.read_csv(csvs[0])
#                     if len(df) > max_rows:
#                         df = df.sample(max_rows, random_state=0)
#                     path = f"{DATA_DIR}/{run_id}_kaggle_{Path(csvs[0]).stem}.csv"
#                     df.to_csv(path, index=False)
#                     meta = {"name": slug, "uri": slug, "license": "unknown", "est_rows": len(df),
#                             "schema": {"features": list(df.columns), "target": None}, "downloaded_to": path}
#                     results.append(meta)
#                     agent_log(run_id, f"[data_agent] Kaggle candidate saved: {path}")
#                 except Exception as e:
#                     agent_log(run_id, f"[data_agent] Kaggle candidate skip {ds.ref}: {e}")
#         except Exception as e:
#             agent_log(run_id, f"[data_agent] Kaggle search error for '{q}': {e}")
#     return results

# # ---------------------
# # DuckDuckGo quick web search + simple CSV / table scraping
# # ---------------------
# def ddg_search_links(query: str, max_results:int=6) -> List[str]:
#     """Perform a lightweight DuckDuckGo HTML search and return hrefs (best-effort)."""
#     try:
#         url = "https://duckduckgo.com/html/"
#         r = requests.post(url, data={"q": query}, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
#         soup = BeautifulSoup(r.text, "html.parser")
#         links = []
#         for a in soup.find_all("a", href=True):
#             href = a["href"]
#             if href.startswith("/l/?kh="):
#                 # ddg redirect -> extract url param 'uddg'
#                 from urllib.parse import parse_qs, urlparse, unquote
#                 qs = parse_qs(urlparse(href).query)
#                 if "uddg" in qs:
#                     links.append(unquote(qs["uddg"][0]))
#                 continue
#             if href.startswith("http"):
#                 links.append(href)
#             if len(links) >= max_results:
#                 break
#         return links
#     except Exception:
#         return []

# def try_scrape_tabular(run_id: str, url: str, max_rows:int=5000) -> Optional[dict]:
#     """
#     Try to fetch URL, handle:
#       - direct CSV (.csv)
#       - HTML table(s) -> convert to pandas
#     Save a CSV locally and return metadata dict on success.
#     """
#     try:
#         agent_log(run_id, f"[data_agent] attempting web scrape: {url}")
#         r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
#         if r.status_code != 200:
#             return None
#         # direct CSV
#         if url.lower().endswith(".csv") or "text/csv" in (r.headers.get("content-type","")):
#             df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd, "compat") else pd.read_csv(io.StringIO(r.text))
#             if len(df) > max_rows:
#                 df = df.sample(max_rows, random_state=0)
#             fname = f"{DATA_DIR}/{run_id}_web_csv.csv"
#             df.to_csv(fname, index=False)
#             return {"name": url, "uri": url, "license":"unknown", "est_rows": len(df),
#                     "schema":{"features": list(df.columns), "target": None}, "downloaded_to": fname}
#         # try parse HTML tables
#         soup = BeautifulSoup(r.text, "html.parser")
#         tables = soup.find_all("table")
#         if not tables:
#             return None
#         for i, tbl in enumerate(tables):
#             try:
#                 df = pd.read_html(str(tbl))[0]
#                 if df.shape[0] == 0 or df.shape[1] == 0:
#                     continue
#                 if len(df) > max_rows:
#                     df = df.sample(max_rows, random_state=0)
#                 fname = f"{DATA_DIR}/{run_id}_web_table_{i}.csv"
#                 df.to_csv(fname, index=False)
#                 return {"name": url, "uri": url, "license": "unknown", "est_rows": len(df),
#                         "schema": {"features": list(df.columns), "target": None}, "downloaded_to": fname}
#             except Exception:
#                 continue
#         return None
#     except Exception as e:
#         agent_log(run_id, f"[data_agent] web scrape failed for {url}: {e}")
#         return None

# def find_on_web(run_id:str, query_list: List[str], max_rows:int=5000) -> List[dict]:
#     results = []
#     for q in query_list:
#         try:
#             links = ddg_search_links(q, max_results=6)
#             for url in links:
#                 meta = try_scrape_tabular(run_id, url, max_rows=max_rows)
#                 if meta:
#                     results.append(meta)
#                     if len(results) >= 3:
#                         return results
#         except Exception as e:
#             agent_log(run_id, f"[data_agent] web search error {q}: {e}")
#             continue
#     return results

# # ---------------------
# # Main orchestration
# # ---------------------
# def get_or_find_dataset(run_id: str, ps: dict, user: dict, min_rows:int=500) -> str:
#     """
#     Priority:
#       1) user-provided upload_path (user['upload_path'])
#       2) search HF datasets (hf_list_datasets + load_dataset)
#       3) search Kaggle (if available)
#       4) web search & scrape (duckduckgo -> CSV/HTML table)
#       5) synthesize dataset (if allowed)
#     Returns local CSV path for selected dataset.
#     """
#     agent_log(run_id, f"[data_agent] get_or_find_dataset start. user keys: {list(user.keys())}")

#     # 1) user provided upload
#     if user and user.get("upload_path"):
#         path = user["upload_path"]
#         if not Path(path).exists():
#             raise FileNotFoundError(f"Provided upload_path not found: {path}")
#         agent_log(run_id, f"[data_agent] using user-provided upload_path: {path}")
#         return path

#     # Build queries using LLM
#     try:
#         q_prompt = _prompt_build_queries(ps)
#         qjson = llm_generate_json(q_prompt) or {}
#         queries = qjson.get("queries", [])[:3]
#     except Exception as e:
#         agent_log(run_id, f"[data_agent] LLM queries failed: {e}")
#         queries = []

#     if not queries:
#         keywords = ps.get("keywords") or [ps.get("domain","")]
#         queries = [ " ".join(keywords) if isinstance(keywords, list) else str(keywords),
#                     ps.get("raw_text",""),
#                     "public tabular dataset " + (ps.get("domain","") or "")
#                   ]
#     queries = [q for q in queries if q and str(q).strip()][:3]
#     agent_log(run_id, f"[data_agent] queries: {queries}")

#     # 2) Hugging Face
#     hf_candidates = find_on_hf(run_id, queries, max_rows=5000)
#     for c in hf_candidates:
#         c["quality_score"] = _score_candidate(c, ps)
#     hf_sorted = sorted(hf_candidates, key=lambda x: x.get("quality_score",0), reverse=True)
#     for cand in hf_sorted:
#         if cand.get("est_rows",0) >= min_rows:
#             agent_log(run_id, f"[data_agent] selected HF candidate {cand['downloaded_to']}")
#             return cand["downloaded_to"]
#     if hf_sorted:
#         agent_log(run_id, f"[data_agent] no HF above min_rows; returning best HF candidate {hf_sorted[0]['downloaded_to']}")
#         return hf_sorted[0]["downloaded_to"]

#     # 3) Kaggle
#     kag_candidates = find_on_kaggle(run_id, queries, max_rows=5000)
#     for c in kag_candidates:
#         c["quality_score"] = _score_candidate(c, ps)
#     kag_sorted = sorted(kag_candidates, key=lambda x: x.get("quality_score",0), reverse=True)
#     for cand in kag_sorted:
#         if cand.get("est_rows",0) >= min_rows:
#             agent_log(run_id, f"[data_agent] selected Kaggle candidate {cand['downloaded_to']}")
#             return cand["downloaded_to"]
#     if kag_sorted:
#         agent_log(run_id, f"[data_agent] returning best Kaggle candidate {kag_sorted[0]['downloaded_to']}")
#         return kag_sorted[0]["downloaded_to"]

#     # 4) Web
#     web_candidates = find_on_web(run_id, queries, max_rows=5000)
#     for c in web_candidates:
#         c["quality_score"] = _score_candidate(c, ps)
#     web_sorted = sorted(web_candidates, key=lambda x: x.get("quality_score",0), reverse=True)
#     for cand in web_sorted:
#         if cand.get("est_rows",0) >= min_rows:
#             agent_log(run_id, f"[data_agent] selected Web candidate {cand['downloaded_to']}")
#             return cand["downloaded_to"]
#     if web_sorted:
#         agent_log(run_id, f"[data_agent] returning best Web candidate {web_sorted[0]['downloaded_to']}")
#         return web_sorted[0]["downloaded_to"]

#     # 5) fallback to synthetic
#     allow_synth = ps.get("plan",{}).get("allow_synthetic", True) or ps.get("constraints",{}).get("allow_synthetic", True) or True
#     if allow_synth:
#         agent_log(run_id, "[data_agent] No public dataset found; generating synthetic dataset.")
#         schema_hint = ps.get("schema_hint") or {"rows": 2000, "columns":[
#             {"name":"age","type":"int","range":[18,80]},
#             {"name":"income","type":"float","range":[1000,100000]},
#             {"name":"utilization","type":"float","range":[0,1]},
#             {"name":"default_flag","type":"binary","imbalance_ratio":0.2}
#         ]}
#         synth_uri = synthesize_dataset(run_id, schema_hint)
#         # If synthesize_dataset returns path or dict, unify
#         if isinstance(synth_uri, dict):
#             return synth_uri.get("dataset_uri")
#         return synth_uri

#     raise RuntimeError("No dataset found and synthetic not allowed.")




# app/agents/data_agent.py
import os, time, io, math
from pathlib import Path
from typing import List
import pandas as pd
import requests
from bs4 import BeautifulSoup
from app.utils.llm_clients import llm_generate_json
from app.utils.run_logger import agent_log
from app.agents.synthetic_agent import synthesize_dataset
from app.storage import ensure_dirs

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"
ensure_dirs()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _prompt_build_queries(ps: dict):
    return f"""
You are a dataset-finding assistant. Produce JSON with a list 'queries' containing up to 3 short search queries relevant to:
Problem statement: {ps.get('raw_text') or ps}
Keywords: {ps.get('keywords') or []}
Return only JSON: {{ "queries": ["...","..."] }}
"""

def _score_candidate(ds_meta: dict, ps: dict) -> float:
    est_rows = ds_meta.get("est_rows") or 0
    has_target = 0
    schema = ds_meta.get("schema") or {}
    features = schema.get("features") or []
    target_name = ps.get("target")
    if target_name and target_name in features:
        has_target = 1
    license_ok = 1 if str(ds_meta.get("license","")).lower().startswith(("cc","mit","apache")) else 0
    score = math.tanh(est_rows/20000.0)*0.6 + 0.3*has_target + 0.1*license_ok
    return float(score)

def find_on_web(run_id: str, queries: List[str], max_rows:int=5000):
    results=[]
    try:
        for q in queries:
            agent_log(run_id, f"[data_agent] ddg query: {q}", agent="data_agent")
            try:
                r = requests.post("https://duckduckgo.com/html/", data={"q": q}, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
                soup = BeautifulSoup(r.text, "html.parser")
                links=[]
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/l/?kh="):
                        from urllib.parse import parse_qs, urlparse, unquote
                        qs = parse_qs(urlparse(href).query)
                        if "uddg" in qs:
                            links.append(unquote(qs["uddg"][0]))
                    elif href.startswith("http"):
                        links.append(href)
                    if len(links)>=6:
                        break
                for url in links:
                    meta = _try_scrape_tabular(run_id, url, max_rows=max_rows)
                    if meta:
                        results.append(meta)
                        if len(results)>=3:
                            return results
            except Exception as e:
                agent_log(run_id, f"[data_agent] web search error for q '{q}': {e}", agent="data_agent")
    except Exception:
        pass
    return results

def _try_scrape_tabular(run_id: str, url: str, max_rows:int=5000):
    try:
        agent_log(run_id, f"[data_agent] attempting web scrape: {url}", agent="data_agent")
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200:
            return None
        ct = r.headers.get("content-type","").lower()
        if url.lower().endswith(".csv") or "text/csv" in ct:
            try:
                txt = r.content.decode("utf-8", errors="replace")
                df = pd.read_csv(io.StringIO(txt))
            except Exception:
                df = pd.read_csv(io.BytesIO(r.content))
            if len(df)>max_rows:
                df = df.sample(max_rows, random_state=0)
            path = f"{DATA_DIR}/web_{int(time.time())}.csv"
            df.to_csv(path, index=False)
            return {"name": url, "uri": url, "license":"unknown", "est_rows": len(df), "schema":{"features": list(df.columns)}, "downloaded_to": path}
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return None
        for i,tbl in enumerate(tables):
            try:
                df_list = pd.read_html(str(tbl))
                if not df_list: continue
                df = df_list[0]
                if df.shape[0]==0: continue
                if len(df)>max_rows: df = df.sample(max_rows, random_state=0)
                path = f"{DATA_DIR}/web_table_{int(time.time())}_{i}.csv"
                df.to_csv(path, index=False)
                return {"name": url, "uri": url, "license":"unknown", "est_rows": len(df), "schema":{"features": list(df.columns)}, "downloaded_to": path}
            except Exception:
                continue
    except Exception as e:
        agent_log(run_id, f"[data_agent] web scrape failed for {url}: {e}", agent="data_agent")
    return None

def get_or_find_dataset(run_id: str, ps: dict, user: dict, min_rows:int=500):
    agent_log(run_id, f"[data_agent] start. user keys: {list(user.keys()) if isinstance(user, dict) else user}", agent="data_agent")
    # 1) user uploaded path
    if user and isinstance(user, dict) and user.get("upload_path"):
        path = user["upload_path"]
        if Path(path).exists():
            agent_log(run_id, f"[data_agent] using user-provided upload_path: {path}", agent="data_agent")
            return path
        else:
            agent_log(run_id, f"[data_agent] provided upload path not found: {path}", agent="data_agent")
    # 2) build queries via LLM
    queries=[]
    try:
        qprompt = _prompt_build_queries(ps)
        qjson = llm_generate_json(qprompt) or {}
        queries = qjson.get("queries", [])[:3]
    except Exception as e:
        agent_log(run_id, f"[data_agent] LLM query generation failed: {e}", agent="data_agent")
    if not queries:
        keywords = ps.get("keywords") or [ps.get("domain","")]
        queries = [" ".join(keywords) if isinstance(keywords,list) else str(keywords), ps.get("raw_text",""), "public tabular dataset " + (ps.get("domain","") or "")]
    queries = [q for q in queries if q and str(q).strip()][:3]
    agent_log(run_id, f"[data_agent] queries: {queries}", agent="data_agent")
    # 3) web search & scrape
    web_candidates = find_on_web(run_id, queries, max_rows=5000)
    for c in web_candidates:
        c["quality_score"] = _score_candidate(c, ps)
    web_sorted = sorted(web_candidates, key=lambda x: x.get("quality_score",0), reverse=True)
    for cand in web_sorted:
        if cand.get("est_rows",0) >= min_rows:
            agent_log(run_id, f"[data_agent] selected web candidate {cand['downloaded_to']}", agent="data_agent")
            return cand["downloaded_to"]
    if web_sorted:
        agent_log(run_id, f"[data_agent] returning best web candidate {web_sorted[0]['downloaded_to']}", agent="data_agent")
        return web_sorted[0]["downloaded_to"]
    # 4) fallback to synthetic
    allow_synth = (ps.get("plan",{}).get("allow_synthetic", True) or ps.get("constraints",{}).get("allow_synthetic", True) or True)
    if allow_synth:
        agent_log(run_id, "[data_agent] No public dataset found; generating synthetic dataset.", agent="data_agent")
        schema_hint = ps.get("schema_hint") or {"rows":2000,"columns":[{"name":"age","type":"int","range":[18,80]},{"name":"income","type":"float","range":[1000,100000]},{"name":"utilization","type":"float","range":[0,1]},{"name":"default_flag","type":"binary","imbalance_ratio":0.2}]}
        path = synthesize_dataset(run_id, schema_hint)
        return path
    raise RuntimeError("No dataset found and synthetic not allowed.")
