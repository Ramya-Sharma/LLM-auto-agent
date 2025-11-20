# # app/agents/prep_agent.py
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import json
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from imblearn.over_sampling import SMOTE
# from app.utils.llm_clients import llm_generate_json

# def guess_target(df, ps):
#     if ps.get("target") and ps["target"] in df.columns:
#         return ps["target"]
#     for c in df.columns[::-1]:
#         if c.lower() in ("target","label","y","default","churn"):
#             return c
#     return df.columns[-1]

# def _ask_llm_for_plan(run_id, df_preview, ps):
#     prompt = f"""
# You are a data preprocessing assistant. Given the columns: {list(df_preview.columns)}, propose a JSON with:
# strategies: missing (median|mode|mean), categorical_encoding (onehot|target), scaling (standard|none), imbalance (smote|none)
# leakage_checks: list of checks to run.
# Return JSON only.
# """
#     return llm_generate_json(prompt)

# def preprocess_dataset(run_id, dataset_path, ps):
#     df = pd.read_csv(dataset_path)
#     target = guess_target(df, ps)
#     if target not in df.columns:
#         raise ValueError("Target not found: " + str(target))
#     llm_plan = _ask_llm_for_plan(run_id, df.head(10), ps) or {}
#     strategies = llm_plan.get("strategies") or {"missing":"median","categorical_encoding":"onehot","scaling":"standard","imbalance":"smote"}
#     leakage_suggestions = llm_plan.get("leakage_checks", ["check_id_like_columns", "check_time_columns"])

#     # Drop likely ID columns (basic heuristic)
#     for c in list(df.columns):
#         if c.lower().endswith("id") or c.lower().startswith("id_"):
#             try:
#                 df = df.drop(columns=[c])
#             except Exception:
#                 pass

#     X = df.drop(columns=[target])
#     y = df[target]
#     numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
#     cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

#     num_transformer = Pipeline([
#         ('imputer', SimpleImputer(strategy='median' if strategies.get("missing","median")=="median" else 'mean')),
#         ('scaler', StandardScaler() if strategies.get("scaling","standard")=="standard" else ('passthrough',))
#     ])
#     if cat_cols:
#         cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
#     else:
#         cat_transformer = None

#     transformers = []
#     if numeric_cols:
#         transformers.append(('num', num_transformer, numeric_cols))
#     if cat_cols and cat_transformer:
#         transformers.append(('cat', cat_transformer, cat_cols))

#     preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

#     X_train_raw, X_test_raw, y_train, y_test = train_test_split(df.drop(columns=[target]), y, test_size=0.2,
#                                                                 stratify=y if ps.get("task_type","classification")=="classification" else None,
#                                                                 random_state=42)
#     preprocessor.fit(X_train_raw, y_train)
#     X_train = preprocessor.transform(X_train_raw)
#     X_test = preprocessor.transform(X_test_raw)

#     if strategies.get("imbalance","smote") == "smote" and ps.get("task_type","classification")=="classification":
#         try:
#             sm = SMOTE(random_state=42)
#             X_train, y_train = sm.fit_resample(X_train, y_train)
#         except Exception:
#             pass

#     os.makedirs("artifacts", exist_ok=True)
#     transformer_path = f"artifacts/{run_id}_transformer.joblib"
#     joblib.dump(preprocessor, transformer_path)

#     np = __import__("numpy")
#     train_path = f"artifacts/{run_id}_train.npz"
#     test_path = f"artifacts/{run_id}_test.npz"
#     np.savez(train_path, X=X_train, y=y_train)
#     np.savez(test_path, X=X_test, y=y_test)

#     report = {"n_rows": len(df), "n_features": int(X.shape[1]) if hasattr(X,'shape') else (len(df.columns)-1), "target": target, "strategies": strategies, "leakage_checks": leakage_suggestions}
#     with open(f"artifacts/{run_id}_prep_report.json","w") as f:
#         json.dump(report, f, indent=2)

#     return {"train_path": train_path, "test_path": test_path, "transformer_path": transformer_path, "report": f"artifacts/{run_id}_prep_report.json", "summary": report}

# # app/agents/prep_agent.py
# import os
# import json
# import joblib
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from imblearn.over_sampling import SMOTE
# from app.utils.llm_clients import llm_generate_json

# ARTIFACT_DIR = "artifacts"
# os.makedirs(ARTIFACT_DIR, exist_ok=True)

# # ----- Helpers -----
# def guess_target(df: pd.DataFrame, ps: Dict) -> str:
#     if ps.get("target") and ps["target"] in df.columns:
#         return ps["target"]
#     for c in df.columns[::-1]:
#         if c.lower() in ("target", "label", "y", "default", "churn"):
#             return c
#     return df.columns[-1]

# def _ask_llm_for_plan(run_id: str, df_preview: pd.DataFrame, ps: Dict) -> Dict:
#     prompt = f"""
# You are a data preprocessing assistant. Given the columns: {list(df_preview.columns)}, propose a JSON with:
# strategies: missing (median|mode|mean), categorical_encoding (onehot|ordinal|target), scaling (standard|none), imbalance (smote|none)
# leakage_checks: list of checks to run.
# Return JSON only.
# """
#     try:
#         return llm_generate_json(prompt) or {}
#     except Exception:
#         return {}

# def _is_datetime_series(s: pd.Series) -> bool:
#     return pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s) and pd.to_datetime(s, errors='coerce').notna().any()

# def _extract_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
#     dt_cols = []
#     new_cols = {}
#     for c in df.columns:
#         if _is_datetime_series(df[c]):
#             try:
#                 ser = pd.to_datetime(df[c], errors='coerce')
#                 new_cols[f"{c}__year"] = ser.dt.year.fillna(0).astype(int)
#                 new_cols[f"{c}__month"] = ser.dt.month.fillna(0).astype(int)
#                 new_cols[f"{c}__day"] = ser.dt.day.fillna(0).astype(int)
#                 # optionally: hour, weekday
#                 dt_cols.append(c)
#             except Exception:
#                 continue
#     if new_cols:
#         df = pd.concat([df.drop(columns=dt_cols), pd.DataFrame(new_cols)], axis=1)
#     return df, dt_cols

# def _choose_cat_strategy(col: pd.Series, plan_choice: str, high_cardinality_threshold: int = 30) -> str:
#     """Return 'onehot' or 'ordinal' (or 'target' if implemented)."""
#     if plan_choice:
#         return plan_choice
#     if col.nunique() <= 10:
#         return "onehot"
#     if col.nunique() > high_cardinality_threshold:
#         return "ordinal"
#     return "onehot"

# # ----- Main preprocessing function -----
# def preprocess_dataset(run_id: str, dataset_path: str, ps: Dict) -> Dict:
#     """
#     Read CSV, detect target, preprocess features, save transformer and train/test npz.
#     Returns dict with train_path, test_path, transformer_path, report, summary.
#     """
#     df = pd.read_csv(dataset_path)
#     target = guess_target(df, ps)
#     if target not in df.columns:
#         raise ValueError(f"Target not found: {target}")

#     # Ask LLM for plan (non-blocking if LLM fails)
#     llm_plan = _ask_llm_for_plan(run_id, df.head(10), ps) or {}
#     strategies = llm_plan.get("strategies") or {"missing": "median", "categorical_encoding": "onehot", "scaling": "standard", "imbalance": "smote"}
#     leakage_suggestions = llm_plan.get("leakage_checks", ["check_id_like_columns", "check_time_columns"])

#     # Basic ID drop heuristic
#     drop_cols = []
#     for c in list(df.columns):
#         if c.lower().endswith("id") or c.lower().startswith("id_"):
#             drop_cols.append(c)
#     if drop_cols:
#         df = df.drop(columns=drop_cols, errors="ignore")

#     # Extract datetime features (if any)
#     df, dt_cols = _extract_datetime_columns(df)

#     # Separate X,y
#     X = df.drop(columns=[target])
#     y = df[target]

#     # Detect types
#     numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
#     # treat booleans as numeric if needed
#     cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
#     # After datetime extraction, some columns may be numeric now; recompute
#     numeric_cols = [c for c in numeric_cols if c in X.columns]
#     cat_cols = [c for c in cat_cols if c in X.columns and c not in numeric_cols]

#     # Build transformers
#     transformers = []
#     feature_name_lists = []

#     # Numeric transformer
#     if numeric_cols:
#         num_impute_strategy = "median" if strategies.get("missing", "median") == "median" else "mean"
#         num_steps = [("imputer", SimpleImputer(strategy=num_impute_strategy))]
#         if strategies.get("scaling", "standard") == "standard":
#             num_steps.append(("scaler", StandardScaler()))
#         num_transformer = Pipeline(num_steps)
#         transformers.append(("num", num_transformer, numeric_cols))
#         feature_name_lists.append(numeric_cols)

#     # Categorical transformer: handle per-column strategy (onehot vs ordinal) to avoid explosion
#     if cat_cols:
#         # Split categorical columns into low-card and high-card lists
#         low_card_cols = []
#         high_card_cols = []
#         for c in cat_cols:
#             strategy_choice = strategies.get("categorical_encoding") or None
#             chosen = _choose_cat_strategy(X[c], strategy_choice)
#             if chosen == "onehot":
#                 low_card_cols.append(c)
#             else:
#                 high_card_cols.append(c)

#         if low_card_cols:
#             cat_pipeline = Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
#             ])
#             transformers.append(("cat_onehot", cat_pipeline, low_card_cols))
#             # feature names for onehot will be expanded later via get_feature_names_out

#         if high_card_cols:
#             # Use OrdinalEncoder (as a simple fallback for high cardinality)
#             ord_pipeline = Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
#             ])
#             transformers.append(("cat_ord", ord_pipeline, high_card_cols))
#             feature_name_lists.append(high_card_cols)

#     # Create ColumnTransformer
#     preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)

#     # Train/test split (stratify if classification)
#     stratify_arg = y if ps.get("task_type", "classification") == "classification" else None
#     X_train_raw, X_test_raw, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=stratify_arg, random_state=42
#     )

#     # Fit transformer on train and transform both
#     preprocessor.fit(X_train_raw, y_train)
#     X_train_t = preprocessor.transform(X_train_raw)
#     X_test_t = preprocessor.transform(X_test_raw)

#     # Ensure numpy arrays (dense) for SMOTE and saving
#     if hasattr(X_train_t, "toarray"):
#         X_train_t = X_train_t.toarray()
#     if hasattr(X_test_t, "toarray"):
#         X_test_t = X_test_t.toarray()

#     # Imbalance handling (SMOTE) - only for classification
#     if strategies.get("imbalance", "smote") == "smote" and ps.get("task_type", "classification") == "classification":
#         try:
#             sm = SMOTE(random_state=42)
#             X_train_t, y_train = sm.fit_resample(X_train_t, y_train)
#         except Exception:
#             # If SMOTE fails (e.g., non-numeric target), continue without oversampling
#             pass

#     # Save transformer and train/test arrays
#     transformer_path = os.path.join(ARTIFACT_DIR, f"{run_id}_transformer.joblib")
#     joblib.dump(preprocessor, transformer_path)

#     train_path = os.path.join(ARTIFACT_DIR, f"{run_id}_train.npz")
#     test_path = os.path.join(ARTIFACT_DIR, f"{run_id}_test.npz")
#     np.savez(train_path, X=X_train_t, y=y_train)
#     np.savez(test_path, X=X_test_t, y=y_test)

#     # Build report with derived info
#     try:
#         n_features_after = X_train_t.shape[1]
#     except Exception:
#         n_features_after = None

#     report = {
#         "n_rows": int(len(df)),
#         "n_features_raw": int(X.shape[1]) if hasattr(X, "shape") else None,
#         "n_features_transformed": int(n_features_after) if n_features_after is not None else None,
#         "target": str(target),
#         "dropped_columns": drop_cols,
#         "datetime_extracted": dt_cols,
#         "numeric_columns": numeric_cols,
#         "categorical_low_cardinality": low_card_cols if cat_cols else [],
#         "categorical_high_cardinality": high_card_cols if cat_cols else [],
#         "strategies": strategies,
#         "leakage_checks_suggested": leakage_suggestions,
#     }
#     report_path = os.path.join(ARTIFACT_DIR, f"{run_id}_prep_report.json")
#     with open(report_path, "w", encoding="utf-8") as f:
#         json.dump(report, f, indent=2)

#     return {
#         "train_path": train_path,
#         "test_path": test_path,
#         "transformer_path": transformer_path,
#         "report": report_path,
#         "summary": report,
#     }

# app/agents/prep_agent.py
import os, json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from app.utils.llm_clients import llm_generate_json
from app.utils.run_logger import agent_log

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _df_preview_stats(df: pd.DataFrame, nrows: int = 50) -> Dict[str, Any]:
    preview = df.head(nrows).to_dict(orient="list")
    col_stats = {}
    for c in df.columns:
        ser = df[c]
        dtype = str(ser.dtype)
        nuniq = int(ser.nunique(dropna=True))
        nnans = int(ser.isna().sum())
        pct_null = float(nnans) / max(1, len(ser))
        sample = ser.dropna().astype(str).unique()[:5].tolist()
        col_stats[c] = {"dtype": dtype, "nunique": nuniq, "nulls": nnans, "pct_null": round(pct_null,4), "sample_values": sample}
    return {"preview_rows": min(len(df), nrows), "columns": col_stats, "sample_preview": preview}

def _build_llm_prompt_for_plan(run_id: str, df: pd.DataFrame, user_ps: Dict) -> str:
    stats = _df_preview_stats(df, nrows=40)
    pref = user_ps or {}
    return f"""
You are a data preprocessing expert. I will give you a dataset column summary and a problem statement context.
Return ONLY a JSON object (no explanation), with keys:
- columns: mapping column_name -> {{ "role":"numeric|categorical|datetime|text|id|ignore", "impute":"median|mean|most_frequent|null", "scale":true|false, "encode":"onehot|ordinal|target|none", "extract":[list datetime parts], "embed":true|false }}
- global: {{ "target": <column-name or null>, "task_type": "classification|regression", "imbalance_handling":"smote|none" }}

Dataset summary:
{json.dumps(stats, indent=2)}

Problem statement / preferences:
{json.dumps(pref, indent=2)}
"""

def _map_plan_to_transformers(plan: Dict[str, Any], df: pd.DataFrame):
    transformers=[]
    onehot_cols=[]
    ord_cols=[]
    num_cols=[]
    drop_cols=[]
    datetime_map={}
    text_cols=[]
    for col,spec in plan.get("columns",{}).items():
        role = spec.get("role")
        if role=="numeric":
            num_cols.append(col)
        elif role=="categorical":
            enc = spec.get("encode","onehot")
            if enc=="onehot":
                onehot_cols.append(col)
            else:
                ord_cols.append(col)
        elif role=="datetime":
            datetime_map[col]=spec.get("extract",["year","month","day"])
        elif role=="text":
            text_cols.append(col)
        elif role in ("id","ignore"):
            drop_cols.append(col)
        else:
            onehot_cols.append(col)
    if num_cols:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("num", num_pipe, num_cols))
    if onehot_cols:
        # sklearn OneHotEncoder in older versions doesn't accept sparse=False in some installations -> be conservative
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore")
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
        transformers.append(("cat_onehot", cat_pipe, onehot_cols))
    if ord_cols:
        ord_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        transformers.append(("cat_ord", ord_pipe, ord_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return preprocessor, datetime_map, drop_cols, text_cols

def _apply_datetime_extraction(df: pd.DataFrame, datetime_map: Dict[str, List[str]]):
    new = df.copy()
    for col,parts in datetime_map.items():
        try:
            ser = pd.to_datetime(new[col], errors="coerce")
            for p in parts:
                if p=="year": new[f"{col}__year"] = ser.dt.year.fillna(0).astype(int)
                if p=="month": new[f"{col}__month"] = ser.dt.month.fillna(0).astype(int)
                if p=="day": new[f"{col}__day"] = ser.dt.day.fillna(0).astype(int)
                if p=="hour": new[f"{col}__hour"] = ser.dt.hour.fillna(0).astype(int)
                if p=="weekday": new[f"{col}__weekday"] = ser.dt.weekday.fillna(0).astype(int)
        except Exception:
            continue
        if col in new.columns:
            new = new.drop(columns=[col], errors="ignore")
    return new

def preprocess_dataset(run_id: str, dataset_path: str, ps: Dict) -> Dict:
    agent_log(run_id, f"[prep_agent] starting on {dataset_path}", agent="prep_agent")
    df = pd.read_csv(dataset_path)
    prompt = _build_llm_prompt_for_plan(run_id, df, ps)
    plan = llm_generate_json(prompt) or {}
    if not plan or "columns" not in plan:
        # fallback plan
        cols={}
        for c in df.columns:
            if c.lower().endswith("id") or c.lower().startswith("id_"):
                cols[c] = {"role":"id"}
            elif pd.api.types.is_numeric_dtype(df[c]):
                cols[c] = {"role":"numeric","impute":"median","scale":True}
            elif pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower():
                cols[c] = {"role":"datetime","extract":["year","month","day"]}
            else:
                cols[c] = {"role":"categorical","encode":"onehot"}
        plan = {"columns": cols, "global":{"target": ps.get("target"), "task_type": ps.get("task_type","classification"), "imbalance_handling":"smote"}}
        agent_log(run_id, "[prep_agent] using fallback plan", agent="prep_agent")
    plan_path = os.path.join(ARTIFACT_DIR, f"{run_id}_llm_prep_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    datetime_map = {c:spec.get("extract",["year","month","day"]) for c,spec in plan.get("columns",{}).items() if spec.get("role")=="datetime"}
    if datetime_map:
        df = _apply_datetime_extraction(df, datetime_map)
    drop_cols = [c for c,spec in plan.get("columns",{}).items() if spec.get("role") in ("id","ignore")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    target = plan.get("global",{}).get("target") or ps.get("target")
    if not target or target not in df.columns:
        candidate=None
        for c in df.columns[::-1]:
            if c.lower() in ("target","label","y","default","churn","price"):
                candidate = c; break
        target = candidate or df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    reduced_plan = {"columns":{c:plan.get("columns",{}).get(c,{}) for c in df.columns if c in plan.get("columns",{})}, "global": plan.get("global",{})}
    preprocessor, datetime_map2, drop_after, text_cols = _map_plan_to_transformers(reduced_plan, X)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                stratify=y if plan.get("global",{}).get("task_type", ps.get("task_type","classification"))=="classification" else None,
                                                                random_state=42)
    preprocessor.fit(X_train_raw, y_train)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    if hasattr(X_train, "toarray"): X_train = X_train.toarray()
    if hasattr(X_test, "toarray"): X_test = X_test.toarray()
    if plan.get("global",{}).get("imbalance_handling","smote")=="smote" and plan.get("global",{}).get("task_type", ps.get("task_type","classification"))=="classification":
        try:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception:
            pass
    transformer_path = os.path.join(ARTIFACT_DIR, f"{run_id}_transformer.joblib")
    joblib.dump(preprocessor, transformer_path)
    train_path = os.path.join(ARTIFACT_DIR, f"{run_id}_train.npz")
    test_path = os.path.join(ARTIFACT_DIR, f"{run_id}_test.npz")
    np.savez(train_path, X=X_train, y=y_train)
    np.savez(test_path, X=X_test, y=y_test)
    report = {
        "n_rows": int(len(df)),
        "n_features_raw": int(X.shape[1]) if hasattr(X, "shape") else None,
        "target": str(target),
        "llm_plan_path": plan_path,
        "llm_plan": plan
    }
    report_path = os.path.join(ARTIFACT_DIR, f"{run_id}_prep_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {"train_path": train_path, "test_path": test_path, "transformer_path": transformer_path, "report": report_path, "summary": report}
