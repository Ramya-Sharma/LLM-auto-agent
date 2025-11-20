# # # app/agents/automl_agent.py
# # import numpy as np
# # import joblib
# # import json
# # from flaml import AutoML
# # from app.utils.llm_clients import llm_generate_json

# # def _ask_llm_for_model_plan(ps):
# #     prompt = f"""
# # You are an ML system designer. Given: {ps.get('raw_text') or ps}, suggest a JSON:
# # {{ "candidate_models": ["xgboost","lightgbm"], "hpo_suggestion": {{ "time_minutes": 10 }} }}
# # Return JSON only.
# # """
# #     return llm_generate_json(prompt)

# # def run_automl(run_id, train_npz_path, ps, preferences):
# #     data = np.load(train_npz_path)
# #     X_train = data["X"]
# #     y_train = data["y"]
# #     plan = _ask_llm_for_model_plan(ps) or {}
# #     candidate_models = plan.get("candidate_models", ["xgboost","random_forest","logistic_regression"])
# #     time_budget = int(preferences.get("training_budget_minutes", 5)) * 60
# #     metric = preferences.get("primary_metric", "f1")
# #     automl = AutoML()
# #     automl_settings = {
# #         "time_budget": max(30, time_budget),
# #         "metric": metric,
# #         "task": "classification" if ps.get("task_type","classification")=="classification" else "regression",
# #         "log_file_name": f"artifacts/{run_id}_flaml.log"
# #     }
# #     estimator_map = {"xgboost":"xgboost","lightgbm":"lgbm","random_forest":"rf","logistic_regression":"lr","catboost":"catboost"}
# #     est_list = []
# #     for m in candidate_models:
# #         if m in estimator_map:
# #             est_list.append(estimator_map[m])
# #     if est_list:
# #         automl_settings["estimator_list"] = est_list
# #     automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
# #     model_path = f"artifacts/{run_id}_best_model.pkl"
# #     joblib.dump(automl.model, model_path)
# #     try:
# #         lb = automl.leaderboard()
# #         lb_json = lb.to_dict()
# #     except Exception:
# #         lb_json = {}
# #     return {"artifact_uri": model_path, "leaderboard": lb_json, "model_path": model_path, "automl_settings": automl_settings}


# # app/agents/automl_agent.py
# """
# Universal AutoML Agent
# - Supports tabular (FLAML), text (sentence-transformers -> FLAML), images (resnet18 features -> FLAML)
# - Data-aware model selection, safe fallbacks and clear artifacts
# """
# import os
# import json
# import time
# import joblib
# import numpy as np
# from typing import Dict, Any
# from flaml import AutoML

# # reduce FLAML logging noise
# import logging
# logging.getLogger("flaml").setLevel(logging.WARNING)
# logging.getLogger("flaml.automl.logger").setLevel(logging.WARNING)


# ARTIFACT_DIR = "artifacts"
# os.makedirs(ARTIFACT_DIR, exist_ok=True)

# # Optional imports (may not be installed). We import lazily where used.
# try:
#     from sentence_transformers import SentenceTransformer
#     _HAS_SENTENCE = True
# except Exception:
#     _HAS_SENTENCE = False

# try:
#     import torch
#     import torchvision
#     from torchvision import transforms
#     from PIL import Image
#     _HAS_TORCH = True
# except Exception:
#     _HAS_TORCH = False

# from app.utils.llm_clients import llm_generate_json

# # ---------------------------
# # Helpers
# # ---------------------------
# def _ask_llm_for_model_plan(ps: Dict[str, Any]) -> Dict[str, Any]:
#     prompt = f"""
# You are an ML architect. Given problem statement: {ps.get('raw_text') or ps}.
# Return JSON with keys: candidate_models (list), hpo_suggestion (e.g. time_minutes).
# If you cannot, return empty JSON.
# """
#     try:
#         return llm_generate_json(prompt) or {}
#     except Exception:
#         return {}

# def _safe_save_json(path, obj):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2)

# def _map_models_to_flaml(candidates):
#     """Map human model names to FLAML estimator_list names (best-effort)."""
#     mapper = {
#         "xgboost": "xgboost",
#         "lightgbm": "lgbm",
#         "lgbm": "lgbm",
#         "random_forest": "rf",
#         "rf": "rf",
#         "catboost": "catboost",
#         "logistic_regression": "lrl2",
#         "logreg": "lrl2",
#         "lrl2": "lrl2",
#         "lrl1": "lrl1",
#         "sgd": "sgd",
#         "svc": "svc",
#         "extra_tree": "extra_tree",
#         "kneighbor": "kneighbor",
#         "histgb": "histgb",
#         "enet": "enet",
#     }
#     out = []
#     for c in (candidates or []):
#         if not isinstance(c, str):
#             continue
#         k = c.strip().lower()
#         if k in mapper:
#             out.append(mapper[k])
#         else:
#             # accept if looks like a flaml name
#             out.append(k)
#     # dedupe and filter
#     seen = set(); res = []
#     for e in out:
#         if e not in seen:
#             res.append(e); seen.add(e)
#     return res

# def _detect_tabular_from_npz(path):
#     try:
#         data = np.load(path)
#         return ("X" in data) and ("y" in data)
#     except Exception:
#         return False

# def _dataset_summary(X: np.ndarray, y: np.ndarray):
#     summary = {
#         "n_rows": int(X.shape[0]) if hasattr(X, "shape") else None,
#         "n_features": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else 1,
#     }
#     try:
#         # imbalance
#         unique, counts = np.unique(y, return_counts=True)
#         if len(counts) > 0:
#             ratios = (counts / counts.sum()).tolist()
#             summary["class_balance"] = dict(zip([str(u) for u in unique.tolist()], ratios))
#     except Exception:
#         pass
#     return summary

# # ---------------------------
# # Modality specific helpers
# # ---------------------------
# def _tabular_automl(X_train, y_train, run_id, ps, preferences, estimator_list=None):
#     """Run FLAML on numeric arrays"""
#     time_budget = int(preferences.get("training_budget_minutes", 5)) * 60
#     time_budget = max(30, time_budget)
#     metric = preferences.get("primary_metric", "f1")
#     task = "classification" if ps.get("task_type", "classification") == "classification" else "regression"

#     automl = AutoML()
#     settings = {
#         "time_budget": time_budget,
#         "metric": metric,
#         "task": task,
#         "log_file_name": f"{ARTIFACT_DIR}/{run_id}_flaml.log",
#     }
#     if estimator_list:
#         settings["estimator_list"] = estimator_list

#     # Try fit with exception handling
#     try:
#         automl.fit(X_train=X_train, y_train=y_train, **settings)
#     except ValueError as e:
#         # save error and retry with conservative default
#         err_path = os.path.join(ARTIFACT_DIR, f"{run_id}_flaml_error.txt")
#         with open(err_path, "a", encoding="utf-8") as f:
#             f.write(str(e) + "\n")
#         settings["estimator_list"] = ["xgboost", "lgbm", "rf"]
#         automl = AutoML()
#         automl.fit(X_train=X_train, y_train=y_train, **settings)
#     # Save model
#     model_path = os.path.join(ARTIFACT_DIR, f"{run_id}_best_model.pkl")
#     try:
#         joblib.dump(automl.model, model_path)
#     except Exception:
#         joblib.dump(automl, model_path)
#     # Leaderboard
#     try:
#         lb = automl.leaderboard()
#         lb_json = lb.to_dict()
#     except Exception:
#         lb_json = {}
#     meta = {
#         "artifact_uri": model_path,
#         "leaderboard": lb_json,
#         "model_path": model_path,
#         "automl_settings": settings,
#     }
#     return meta

# def _text_to_embeddings(csv_path, text_column="text", max_samples=None):
#     if not _HAS_SENTENCE:
#         raise RuntimeError("sentence-transformers not installed. Install 'sentence-transformers' to handle text modality.")
#     df = None
#     try:
#         df = __import__("pandas").read_csv(csv_path)
#     except Exception:
#         raise
#     if text_column not in df.columns:
#         # Try find a probable text column
#         candidates = [c for c in df.columns if "text" in c.lower() or "review" in c.lower() or "comment" in c.lower()]
#         if candidates:
#             text_column = candidates[0]
#         else:
#             # use first object dtype column
#             obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
#             if not obj_cols:
#                 raise RuntimeError("No text column found for text modality.")
#             text_column = obj_cols[0]
#     texts = df[text_column].astype(str).tolist()
#     if max_samples and len(texts) > max_samples:
#         texts = texts[:max_samples]
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(texts, show_progress_bar=False)
#     # try to find labels if present
#     label = None
#     for c in ["label", "target", "y"]:
#         if c in df.columns:
#             label = df[c].values
#             break
#     return np.asarray(embeddings), label

# def _image_paths_to_features(paths, max_samples=None):
#     if not _HAS_TORCH:
#         raise RuntimeError("torch/torchvision not installed. Install torch to handle image modality.")
#     if isinstance(paths, str):
#         # assume it's a CSV with image paths
#         try:
#             import pandas as pd
#             df = pd.read_csv(paths)
#             # look for column with path-like values
#             candidates = [c for c in df.columns if any(ext in str(df[c].dropna().astype(str).iloc[0]).lower() for ext in [".jpg", ".png", ".jpeg", ".bmp"])]
#             if candidates:
#                 img_col = candidates[0]
#                 paths_list = df[img_col].tolist()
#             else:
#                 # if the CSV itself is a single column of paths
#                 paths_list = df.iloc[:,0].astype(str).tolist()
#         except Exception as e:
#             raise RuntimeError(f"Unable to read image paths CSV: {e}")
#     elif isinstance(paths, list):
#         paths_list = paths
#     else:
#         raise RuntimeError("Unsupported type for image paths input.")

#     if max_samples:
#         paths_list = paths_list[:max_samples]

#     # feature extractor: resnet18 pretrained, remove last layer
#     device = "cuda" if (torch.cuda.is_available()) else "cpu"
#     resnet = torchvision.models.resnet18(pretrained=True)
#     resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
#     resnet.eval()
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])
#     feats = []
#     for p in paths_list:
#         try:
#             img = Image.open(p).convert("RGB")
#             t = preprocess(img).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 out = resnet(t).cpu().numpy().reshape(-1)
#             feats.append(out)
#         except Exception:
#             # skip unreadable images
#             continue
#     if not feats:
#         raise RuntimeError("No image features could be extracted.")
#     return np.vstack(feats), None

# # ---------------------------
# # Main entrypoint
# # ---------------------------
# def run_automl(run_id: str, train_path: str, ps: Dict[str, Any], preferences: Dict[str, Any]):
#     """
#     Universal orchestrator entry.
#     train_path: for tabular, a .npz with X/y; for text: path to CSV; for image: CSV of paths or list
#     ps: problem statement dict (may include required_modalities)
#     preferences: includes training_budget_minutes, primary_metric, etc.
#     """
#     start = time.time()
#     modality = None
#     # 1) Determine modality from PS if present
#     plan = ps.get("plan") or {}
#     required = plan.get("required_modalities") or ps.get("required_modalities")
#     if required:
#         modality = required[0].lower()

#     # 2) Fallback: detect tabular npz
#     if modality is None:
#         if train_path and isinstance(train_path, str) and train_path.endswith(".npz") and _detect_tabular_from_npz(train_path):
#             modality = "tabular"
#         elif train_path and isinstance(train_path, str) and train_path.endswith(".csv"):
#             # could be text or tabular; choose tabular by default but let LLM suggest
#             modality = "tabular"
#         else:
#             modality = "tabular"

#     # Ask LLM for candidate models (text-only suggestion)
#     llm_plan = _ask_llm_for_model_plan(ps)
#     candidate_models = llm_plan.get("candidate_models") or preferences.get("candidate_models") or []
#     est_list = _map_models_to_flaml(candidate_models) if modality == "tabular" else None

#     meta = {"run_id": run_id, "modality": modality, "candidate_models": candidate_models, "est_list": est_list}

#     try:
#         if modality == "tabular":
#             if not train_path or not train_path.endswith(".npz"):
#                 raise RuntimeError("Tabular modality expects a .npz file with X and y created by preprocess agent.")
#             data = np.load(train_path)
#             X = data["X"]
#             y = data["y"]
#             summary = _dataset_summary(X, y)
#             meta["data_summary"] = summary
#             # data-aware model selection: adjust estimator list based on dataset
#             if not est_list:
#                 # pick based on rows/features/imbalance
#                 n = summary.get("n_rows", 0) or 0
#                 f = summary.get("n_features", 0) or 0
#                 cb = summary.get("class_balance", {})
#                 # heuristics
#                 if n > 50000 or f > 1000:
#                     est_list = ["lgbm", "xgboost"]
#                 elif f > 200:
#                     est_list = ["lgbm", "rf"]
#                 elif any(v < 0.05 for v in (cb.values() if cb else [1])):
#                     est_list = ["xgboost", "catboost"]
#                 else:
#                     est_list = ["xgboost", "lgbm", "rf"]
#             res = _tabular_automl(X, y, run_id, ps, preferences, estimator_list=est_list)
#             meta.update(res)
#         elif modality == "text":
#             # train_path expected as CSV with text column and optional label
#             emb, label = _text_to_embeddings(train_path)
#             if label is None:
#                 raise RuntimeError("No label column found in text CSV; supervised training requires labels.")
#             summary = _dataset_summary(emb, label)
#             meta["data_summary"] = summary
#             # map models (we will use simple linear models or xgboost on embeddings)
#             if not est_list:
#                 est_list = ["lrl2", "xgboost"]
#             res = _tabular_automl(emb, label, run_id, ps, preferences, estimator_list=est_list)
#             meta.update(res)
#         elif modality == "image":
#             feats, label = _image_paths_to_features(train_path)
#             if label is not None:
#                 # ensure alignment; else user can pass labels separately
#                 pass
#             summary = _dataset_summary(feats, label if label is not None else np.zeros(feats.shape[0]))
#             meta["data_summary"] = summary
#             if not est_list:
#                 est_list = ["xgboost", "rf"]
#             res = _tabular_automl(feats, label if label is not None else np.zeros(feats.shape[0]), run_id, ps, preferences, estimator_list=est_list)
#             meta.update(res)
#         else:
#             raise RuntimeError(f"Unsupported modality: {modality}")

#         # Save meta and return
#         meta_path = os.path.join(ARTIFACT_DIR, f"{run_id}_train_meta.json")
#         _safe_save_json(meta_path, meta)
#         meta["meta_path"] = meta_path
#         return meta

#     except Exception as e:
#         # Write error artifact
#         err_path = os.path.join(ARTIFACT_DIR, f"{run_id}_automl_error.txt")
#         with open(err_path, "w", encoding="utf-8") as f:
#             f.write(str(e) + "\n")
#         raise





# # app/agents/automl_agent.py

# import numpy as np
# import joblib
# import json
# import logging
# import traceback
# from flaml import AutoML
# from app.utils.llm_clients import llm_generate_json
# from app.utils.run_logger import agent_log

# # reduce FLAML logging noise
# logging.getLogger("flaml").setLevel(logging.WARNING)
# logging.getLogger("flaml.automl.logger").setLevel(logging.WARNING)


# # ---------------------------------------------------------------------
# # FLAML Supported Estimators (clean, task-separated)
# # ---------------------------------------------------------------------
# FLAML_SUPPORTED = {
#     "classification": {
#         "xgboost", "xgb_limitdepth", "rf", "lgbm", "catboost",
#         "extra_tree", "kneighbor", "svc", "sgd", "histgb"
#     },
#     "regression": {
#         "xgboost", "rf", "lgbm", "extra_tree", "histgb", "sgd"
#     }
# }

# # Friendly → FLAML name mapping
# ESTIMATOR_MAP = {
#     "xgboost": "xgboost",
#     "lightgbm": "lgbm",
#     "lgbm": "lgbm",
#     "random_forest": "rf",
#     "rf": "rf",
#     "catboost": "catboost",
#     "logistic_regression": "lrl1",
#     "logreg": "lrl1",
#     "lr": "lrl1",
#     "sgd": "sgd",
#     "svc": "svc",
#     "kneighbor": "kneighbor",
#     "extra_tree": "extra_tree"
# }


# # ---------------------------------------------------------------------
# # LLM decides model list & task type
# # ---------------------------------------------------------------------
# def _ask_llm_plan(ps, X_train, y_train):
#     """
#     LLM decides:
#       - task_type: classification or regression
#       - candidate_models: ["xgboost", "lgbm"]
#       - time budget hint (optional)
#     """
#     prompt = f"""
# You are an expert ML system designer.

# Given this problem statement and dataset:
# Problem Statement: {ps}
# y (target) preview (dtype={str(y_train.dtype)}): {y_train[:10]}

# Decide the correct ML task type based on the nature of y:
# - "classification" → discrete labels, categories, small number of unique values.
# - "regression" → continuous numeric values.

# Return STRICT JSON:
# {{
#   "task_type": "classification" or "regression",
#   "candidate_models": ["xgboost","lightgbm","random_forest"],
#   "hpo_suggestion": {{"time_minutes": 10}}
# }}
# Only return JSON.
# """

#     try:
#         return llm_generate_json(prompt) or {}
#     except Exception:
#         return {}


# # ---------------------------------------------------------------------
# # Clean estimator sanitation (no if/else on task)
# # ---------------------------------------------------------------------
# def _sanitize_candidate_models(candidate_models, task):
#     task = task if task in ("classification", "regression") else "classification"

#     out = []
#     for m in candidate_models:
#         m_low = (m or "").lower()

#         mapped = ESTIMATOR_MAP.get(m_low, None)

#         # if not mapped but already valid FLAML name
#         if not mapped and m_low in FLAML_SUPPORTED[task]:
#             mapped = m_low

#         # keep only valid models for the task
#         if mapped and mapped in FLAML_SUPPORTED[task]:
#             out.append(mapped)

#     # fallback
#     if not out:
#         out = ["xgboost", "lgbm", "rf"]

#     # dedupe
#     seen = set()
#     res = []
#     for e in out:
#         if e not in seen:
#             seen.add(e)
#             res.append(e)
#     return res


# # ---------------------------------------------------------------------
# # Main AutoML Runner
# # ---------------------------------------------------------------------
# def run_automl(run_id, train_npz_path, ps, preferences):

#     # Load data
#     try:
#         data = np.load(train_npz_path)
#         X_train = data["X"]
#         y_train = data["y"]
#     except Exception as e:
#         agent_log(run_id, f"[automl_agent] failed loading {train_npz_path}: {e}")
#         raise

#     y_train = np.asarray(y_train).ravel()
#     if y_train is None:
#         raise ValueError("y_train is None")

#     # LLM decides everything
#     llm_plan = _ask_llm_plan(ps, X_train, y_train)

#     task_type = llm_plan.get("task_type", "classification")
#     candidate_models = llm_plan.get(
#         "candidate_models",
#         ["xgboost", "lightgbm", "random_forest"]
#     )

#     est_list = _sanitize_candidate_models(candidate_models, task_type)

#     # Time & metric
#     time_budget = int(preferences.get("training_budget_minutes", 5)) * 60
#     metric = preferences.get("primary_metric", "f1")

#     # AutoML setup
#     automl = AutoML()
#     settings = {
#         "time_budget": max(30, time_budget),
#         "metric": metric,
#         "task": task_type,
#         "estimator_list": est_list,
#         "log_file_name": f"artifacts/{run_id}_flaml.log"
#     }

#     agent_log(run_id,
#               f"[automl_agent] AutoML Starting | task={task_type} | "
#               f"metric={metric} | time={settings['time_budget']}s | "
#               f"models={est_list}")

#     # Run AutoML
#     try:
#         automl.fit(X_train=X_train, y_train=y_train, **settings)
#     except Exception as e:
#         tb = traceback.format_exc()
#         agent_log(run_id, f"[automl_agent] AutoML fit failed: {e}\n{tb}")
#         raise

#     # Save best model
#     model_path = f"artifacts/{run_id}_best_model.pkl"
#     try:
#         joblib.dump(automl.model, model_path)
#     except Exception:
#         try:
#             joblib.dump(automl, model_path)
#             agent_log(run_id,
#                       "[automl_agent] Pickled entire AutoML object as fallback")
#         except Exception as e:
#             agent_log(run_id, f"[automl_agent] model save failed: {e}")

#     # Leaderboard extraction
#     try:
#         lb = automl.leaderboard()
#         lb_json = lb.to_dict()
#     except Exception:
#         lb_json = {}

#     res = {
#         "artifact_uri": model_path,
#         "leaderboard": lb_json,
#         "model_path": model_path,
#         "automl_settings": settings
#     }
#     agent_log(run_id, "[automl_agent] Finished AutoML run successfully.")

#     return res




# app/agents/automl_agent.py
import os, json, traceback
import numpy as np
import joblib
from flaml import AutoML
from app.utils.llm_clients import llm_generate_json
from app.utils.run_logger import agent_log

FLAML_SUPPORTED = {
    "classification": {"xgboost","xgb_limitdepth","rf","lgbm","catboost","extra_tree","kneighbor","svc","sgd","histgb"},
    "regression": {"xgboost","rf","lgbm","extra_tree","histgb","sgd"}
}
ESTIMATOR_MAP = {"xgboost":"xgboost","lightgbm":"lgbm","lgbm":"lgbm","random_forest":"rf","rf":"rf","catboost":"catboost","logistic_regression":"lrl1","logreg":"lrl1","lr":"lrl1","sgd":"sgd","svc":"svc","kneighbor":"kneighbor","extra_tree":"extra_tree"}

def _ask_llm_plan(ps, X_train, y_train):
    prompt = f"""
You are an ML system designer.
Problem Statement: {ps.get('raw_text') or ps}
y (preview): {list(y_train[:10])}
Decide task_type and candidate_models. Return JSON:
{{ "task_type": "classification" or "regression", "candidate_models":["xgboost","lgbm"], "hpo_suggestion":{{"time_minutes": 10}} }}
Only return JSON.
"""
    try:
        return llm_generate_json(prompt) or {}
    except Exception:
        return {}

def _sanitize_candidate_models(candidate_models, task):
    task = task if task in ("classification","regression") else "classification"
    out=[]
    for m in candidate_models:
        m_low = (m or "").lower()
        mapped = ESTIMATOR_MAP.get(m_low, None)
        if not mapped and m_low in FLAML_SUPPORTED[task]:
            mapped = m_low
        if mapped and mapped in FLAML_SUPPORTED[task]:
            out.append(mapped)
    if not out:
        out = ["xgboost","lgbm","rf"]
    # dedupe
    seen=set(); res=[]
    for e in out:
        if e not in seen:
            seen.add(e); res.append(e)
    return res

def run_automl(run_id: str, train_npz_path: str, ps: dict, preferences: dict):
    agent_log(run_id, f"[automl_agent] starting with train {train_npz_path}", agent="automl_agent")
    try:
        data = np.load(train_npz_path, allow_pickle=True)
        X_train = data["X"]
        y_train = data["y"]
    except Exception as e:
        raise RuntimeError(f"Failed to load train npz: {e}")
    y_train = np.asarray(y_train).ravel()
    llm_plan = _ask_llm_plan(ps, X_train, y_train)
    task_type = llm_plan.get("task_type", ps.get("task_type", "classification"))
    candidate_models = llm_plan.get("candidate_models", ["xgboost","lgbm","rf"])
    est_list = _sanitize_candidate_models(candidate_models, task_type)
    time_budget = int(preferences.get("training_budget_minutes", 5)) * 60
    metric = preferences.get("primary_metric", "f1")
    automl = AutoML()
    settings = {
        "time_budget": max(30, time_budget),
        "metric": metric,
        "task": task_type,
        "estimator_list": est_list,
        "log_file_name": f"artifacts/{run_id}_flaml.log"
    }
    agent_log(run_id, f"[automl_agent] settings: {settings}", agent="automl_agent")
    try:
        automl.fit(X_train=X_train, y_train=y_train, **settings)
    except Exception as e:
        tb = traceback.format_exc()
        agent_log(run_id, f"[automl_agent] AutoML fit failed: {e}\n{tb}", agent="automl_agent")
        raise
    model_path = f"artifacts/{run_id}_best_model.pkl"
    try:
        joblib.dump(automl.model, model_path)
    except Exception:
        try:
            joblib.dump(automl, model_path)
        except Exception as e:
            agent_log(run_id, f"[automl_agent] model save failed: {e}", agent="automl_agent")
    try:
        lb = automl.leaderboard()
        lb_json = lb.to_dict()
    except Exception:
        lb_json = {}
    res = {"artifact_uri": model_path, "leaderboard": lb_json, "model_path": model_path, "automl_settings": settings, "task_type": task_type}
    agent_log(run_id, "[automl_agent] finished", agent="automl_agent")
    return res
