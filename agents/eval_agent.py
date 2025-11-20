# # app/agents/eval_agent.py
# import numpy as np
# import joblib
# import json
# from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
# import matplotlib.pyplot as plt
# from app.utils.llm_clients import llm_generate_json

# def evaluate_model(run_id, test_npz_path, model_path, transformer_path, ps):
#     data = np.load(test_npz_path)
#     X_test = data["X"]
#     y_test = data["y"]
#     model = joblib.load(model_path)
#     y_pred = model.predict(X_test)
#     y_prob = None
#     try:
#         if hasattr(model, "predict_proba"):
#             y_prob = model.predict_proba(X_test)[:,1]
#     except Exception:
#         y_prob = None

#     metrics = {"f1": float(f1_score(y_test, y_pred))}
#     try:
#         if y_prob is not None:
#             metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
#     except Exception:
#         metrics["roc_auc"] = None
#     metrics["precision"] = float(precision_score(y_test, y_pred))
#     metrics["recall"] = float(recall_score(y_test, y_pred))

#     # plots
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest')
#     plt.title('confusion')
#     plt.colorbar()
#     plt.savefig(f"artifacts/{run_id}_confusion.png")
#     plt.close()

#     if y_prob is not None:
#         from sklearn.metrics import roc_curve
#         fpr, tpr, _ = roc_curve(y_test, y_prob)
#         plt.figure()
#         plt.plot(fpr, tpr)
#         plt.title("ROC")
#         plt.savefig(f"artifacts/{run_id}_roc.png")
#         plt.close()

#     prompt = f"""
# You are an ML evaluator. Given these metrics: {json.dumps(metrics)}, and problem statement: {ps.get('raw_text') or ps},
# 1) Provide short interpretation of strengths and weaknesses.
# 2) Recommend whether precision or recall should be prioritized and why.
# 3) Produce a model card JSON: {{ model_name, metrics, recommended_use, limitations, next_steps }}.
# Return JSON only.
# """
#     card = llm_generate_json(prompt)
#     if not card:
#         card = {"model_name": str(model.__class__.__name__), "metrics": metrics, "recommended_use": "Use if F1 acceptable", "limitations": [], "next_steps": ["Collect more data", "Tune threshold"]}

#     with open(f"artifacts/{run_id}_evaluation.json","w") as f:
#         json.dump({"metrics": metrics, "model_card": card}, f, indent=2)

#     return {"best_model": card.get("model_name", str(model.__class__.__name__)), "metrics": metrics, "plots":{"confusion":f"artifacts/{run_id}_confusion.png"}, "model_card": card}


# # app/agents/eval_agent.py
# import numpy as np
# import joblib
# import json
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import (
#     f1_score,
#     roc_auc_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
#     roc_curve,
#     auc
# )
# from sklearn.preprocessing import label_binarize
# from sklearn.calibration import calibration_curve
# from app.utils.llm_clients import llm_generate_json


# def _get_scores(model, X):
#     """
#     Returns prediction scores:
#     - predict_proba -> probability
#     - decision_function -> continuous score
#     - else -> None
#     """
#     if hasattr(model, "predict_proba"):
#         try:
#             proba = model.predict_proba(X)
#             # binary: return N x 1 vector
#             if proba.shape[1] == 2:
#                 return proba[:, 1]
#             return proba
#         except:
#             pass

#     if hasattr(model, "decision_function"):
#         try:
#             score = model.decision_function(X)
#             return score
#         except:
#             pass

#     return None


# def _plot_confusion_matrix(cm, classes, run_id):
#     plt.figure(figsize=(5, 4))
#     plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix")
#     plt.colorbar()

#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     plt.xlabel("Predicted label")
#     plt.ylabel("True label")
#     plt.tight_layout()
#     path = f"artifacts/{run_id}_confusion.png"
#     plt.savefig(path)
#     plt.close()
#     return path


# def _plot_roc_multiclass(y_test, scores, classes, run_id):
#     """
#     Supports binary or multiclass ROC plotting.
#     """
#     plt.figure(figsize=(6, 5))

#     if len(classes) == 2:
#         # Scores is either vector (binary) or probability array
#         if scores.ndim > 1:
#             y_score = scores[:, 1]
#         else:
#             y_score = scores

#         fpr, tpr, _ = roc_curve(y_test, y_score)
#         plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc(fpr, tpr):.3f})")
#     else:
#         # Multiclass ROC
#         y_test_bin = label_binarize(y_test, classes=classes)
#         for i, cls in enumerate(classes):
#             try:
#                 fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores[:, i])
#                 plt.plot(fpr, tpr, label=f"Class {cls} AUC={auc(fpr, tpr):.3f}")
#             except:
#                 continue

#     plt.plot([0, 1], [0, 1], "k--")
#     plt.title("ROC Curve")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend()

#     path = f"artifacts/{run_id}_roc.png"
#     plt.savefig(path)
#     plt.close()
#     return path


# def _plot_calibration_curve(y_test, scores, run_id):
#     """
#     Calibration curve: reliability diagram.
#     Only for binary classification.
#     """
#     try:
#         prob_pos = scores
#         frac_pos, mean_pred = calibration_curve(y_test, prob_pos, n_bins=10)
#     except:
#         return None

#     plt.figure(figsize=(5, 5))
#     plt.plot(mean_pred, frac_pos, "s-", label="Model")
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.xlabel("Predicted probability")
#     plt.ylabel("Fraction of positives")
#     plt.title("Calibration Curve")
#     plt.legend()

#     path = f"artifacts/{run_id}_calibration.png"
#     plt.savefig(path)
#     plt.close()
#     return path


# def evaluate_model(run_id, test_npz_path, model_path, transformer_path, ps):
#     # ===== LOAD DATA =====
#     data = np.load(test_npz_path)
#     X_test = data["X"]
#     y_test = data["y"]

#     # Load transformer (important!)
#     transformer = joblib.load(transformer_path)

#     # Load model
#     model = joblib.load(model_path)

#     # ===== TRANSFORM X =====
#     try:
#         X_test = transformer.transform(X_test)
#     except:
#         pass  # Already transformed

#     # ===== PREDICT =====
#     y_pred = model.predict(X_test)
#     scores = _get_scores(model, X_test)

#     # ===== METRICS =====
#     classes = np.unique(y_test)

#     metrics = {
#         "f1": float(f1_score(y_test, y_pred, average="weighted")),
#         "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
#         "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
#     }

#     # ROC-AUC if possible
#     try:
#         if scores is not None:
#             if len(classes) == 2:
#                 auc_value = roc_auc_score(y_test, scores)
#             else:
#                 auc_value = roc_auc_score(y_test, scores, multi_class="ovr")
#             metrics["roc_auc"] = float(auc_value)
#     except:
#         metrics["roc_auc"] = None

#     # ===== SAVE PREDICTIONS =====
#     pred_df = pd.DataFrame({
#         "y_true": y_test,
#         "y_pred": y_pred,
#         "score": scores if scores is not None else [None]*len(y_test)
#     })
#     pred_path = f"artifacts/{run_id}_predictions.csv"
#     pred_df.to_csv(pred_path, index=False)

#     # ===== PLOTS =====
#     cm = confusion_matrix(y_test, y_pred)
#     cm_path = _plot_confusion_matrix(cm, classes, run_id)

#     roc_path = None
#     if scores is not None:
#         roc_path = _plot_roc_multiclass(y_test, scores, classes, run_id)

#     calibration_path = None
#     if scores is not None and len(classes) == 2:
#         calibration_path = _plot_calibration_curve(y_test, scores, run_id)

#     # ===== MODEL CARD (LLM) =====
#     prompt = f"""
# You are an ML evaluator. Given metrics: {json.dumps(metrics)},
# problem: {ps.get('raw_text') or ps},
# return ONLY JSON:
# {{
#   "model_name": "",
#   "metrics": {{}},
#   "strengths": [],
#   "weaknesses": [],
#   "recommended_use": "",
#   "limitations": [],
#   "next_steps": []
# }}
# """
#     card = llm_generate_json(prompt)

#     if not card:
#         card = {
#             "model_name": str(model.__class__.__name__),
#             "metrics": metrics,
#             "strengths": ["Model evaluation completed."],
#             "weaknesses": [],
#             "recommended_use": "General classification",
#             "limitations": ["LLM could not generate card"],
#             "next_steps": ["Improve dataset", "Collect more samples"],
#         }

#     # ===== SAVE EVALUATION REPORT =====
#     eval_path = f"artifacts/{run_id}_evaluation.json"
#     with open(eval_path, "w") as f:
#         json.dump({"metrics": metrics, "model_card": card}, f, indent=2)

#     return {
#         "best_model": card.get("model_name", str(model.__class__.__name__)),
#         "metrics": metrics,
#         "plots": {
#             "confusion": cm_path,
#             "roc": roc_path,
#             "calibration": calibration_path
#         },
#         "predictions_file": pred_path,
#         "model_card": card
#     }


# app/agents/eval_agent.py
import os, json, traceback
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from app.utils.llm_clients import llm_generate_json
from app.utils.run_logger import agent_log

ARTIFACT_DIR="artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _safe_save_fig(run_id, fig_name):
    path = os.path.join(ARTIFACT_DIR, f"{run_id}_{fig_name}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _get_model_scores(model, X):
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            if p is None: return None, False, False
            if p.ndim==2 and p.shape[1]==2:
                return p[:,1], True, False
            return p, True, True
    except Exception:
        pass
    try:
        if hasattr(model, "decision_function"):
            score = model.decision_function(X)
            return score, False, False
    except Exception:
        pass
    return None, False, False

def evaluate_model(run_id, test_npz_path, model_path, transformer_path, ps):
    agent_log(run_id, f"[eval_agent] loading test {test_npz_path} model {model_path}", agent="eval_agent")
    try:
        data = np.load(test_npz_path, allow_pickle=True)
        X_test = data["X"]
        y_test = data["y"]
    except Exception as e:
        raise RuntimeError(f"Failed to load test data: {e}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    transformer = None
    try:
        transformer = joblib.load(transformer_path)
    except Exception:
        transformer = None
    X_trans = X_test
    try:
        if transformer is not None:
            X_trans = transformer.transform(X_test)
    except Exception:
        pass
    try:
        y_pred = model.predict(X_trans)
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")
    scores, prob_flag, multiclass_prob = _get_model_scores(model, X_trans)
    task = ps.get("task_type")
    if task is None:
        if np.asarray(y_test).dtype.kind in "ifu":
            task = "regression"
        else:
            task = "classification"
    metrics={}
    try:
        if task=="regression":
            metrics["mse"] = float(mean_squared_error(y_test, y_pred))
            metrics["r2"] = float(r2_score(y_test, y_pred))
        else:
            avg="weighted"
            metrics["f1"]=float(f1_score(y_test, y_pred, average=avg, zero_division=0))
            metrics["precision"]=float(precision_score(y_test, y_pred, average=avg, zero_division=0))
            metrics["recall"]=float(recall_score(y_test, y_pred, average=avg, zero_division=0))
            if scores is not None:
                try:
                    if len(np.unique(y_test))==2 and (not multiclass_prob):
                        metrics["roc_auc"] = float(auc(*roc_curve(y_test, scores)[:2]))
                except Exception:
                    pass
    except Exception:
        pass
    # Save predictions
    try:
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        if scores is not None and (not hasattr(scores, "ndim") or scores.ndim==1):
            pred_df["score"] = list(scores)
        pred_path = os.path.join(ARTIFACT_DIR, f"{run_id}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
    except Exception:
        pred_path = None
    # minimal plots
    plots={}
    try:
        # confusion if classification
        if task!="regression":
            cm = confusion_matrix(y_test, y_pred)
            fig = plt.figure(figsize=(4,3)); plt.imshow(cm); plt.colorbar(); plt.title("Confusion"); plt.xlabel("pred"); plt.ylabel("true")
            plots["confusion"] = _safe_save_fig(run_id, "confusion")
    except Exception:
        pass
    # LLM model card
    try:
        prompt = f"""
You are an ML evaluator. Given metrics: {json.dumps(metrics)},
problem: {ps.get('raw_text') or ps},
return ONLY JSON:
{{ "model_name": "", "metrics": {{}}, "strengths": [], "weaknesses": [], "recommended_use":"", "limitations": [], "next_steps": [] }}
"""
        card = llm_generate_json(prompt) or {}
    except Exception:
        card = None
    if not card:
        card = {"model_name": str(type(model).__name__), "metrics": metrics, "strengths":["Evaluation completed."], "weaknesses":[], "recommended_use":"Refer to metrics", "limitations":[], "next_steps":["Collect more data","Tune hyperparameters"]}
    eval_record = {"metrics": metrics, "model_card": card, "plots": plots, "predictions": pred_path}
    eval_path = os.path.join(ARTIFACT_DIR, f"{run_id}_evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2, default=str)
    return {"best_model": card.get("model_name", ""), "metrics": metrics, "plots": plots, "predictions_file": pred_path, "model_card": card}
