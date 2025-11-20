# # app/agents/deploy_agent.py
# from app.utils.llm_clients import llm_generate_json
# import json
# import os

# def generate_deploy_scaffold(run_id, model_path, transformer_path, request_schema):
#     prompt = f"""
# You are a deployment template generator. Produce JSON:
# {{ "serve_py": "...", "dockerfile": "..." }}
# The serve_py should be a FastAPI app that loads model at '{model_path}' and transformer at '{transformer_path}', exposes POST /predict and validates request against schema {json.dumps(request_schema)}.
# Return JSON only.
# """
#     j = llm_generate_json(prompt)
#     if j and "serve_py" in j and "dockerfile" in j:
#         serve = j["serve_py"]
#         dockerfile = j["dockerfile"]
#     else:
#         # fallback safe templates
#         serve = f'''
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib, pandas as pd

# class Req(BaseModel):
# {chr(10).join([f"    {k}: {('int' if v=='int' else 'float')}" for k,v in request_schema.items()])}

# app = FastAPI()
# model = joblib.load("{model_path}")
# transformer = joblib.load("{transformer_path}")

# @app.post("/predict")
# def predict(payload: Req):
#     df = pd.DataFrame([payload.dict()])
#     X = transformer.transform(df)
#     pred = model.predict(X)
#     return {{"prediction": int(pred[0])}}
# '''
#         dockerfile = f'''
# FROM python:3.10-slim
# WORKDIR /app
# COPY artifacts /app/artifacts
# RUN pip install fastapi uvicorn joblib pandas scikit-learn
# CMD ["uvicorn","artifacts.{run_id}_serve:app","--host","0.0.0.0","--port","8080"]
# '''
#     os.makedirs("artifacts", exist_ok=True)
#     serve_path = f"artifacts/{run_id}_serve.py"
#     docker_path = f"artifacts/{run_id}_Dockerfile"
#     with open(serve_path, "w") as f:
#         f.write(serve)
#     with open(docker_path, "w") as f:
#         f.write(dockerfile)
#     return {"serve": serve_path, "dockerfile": docker_path}

# app/agents/deploy_agent.py
import os, json
from app.utils.llm_clients import llm_generate_json

def generate_deploy_scaffold(run_id, model_path, transformer_path, request_schema):
    prompt = f"""
You are a deployment template generator. Produce JSON:
{{ "serve_py": "...", "dockerfile": "..." }}
The serve_py should be a FastAPI app that loads model at '{model_path}' and transformer at '{transformer_path}', exposes POST /predict and validates request against schema {json.dumps(request_schema)}.
Return JSON only.
"""
    j = llm_generate_json(prompt)
    if j and isinstance(j, dict) and "serve_py" in j and "dockerfile" in j:
        serve = j["serve_py"]; dockerfile = j["dockerfile"]
    else:
        serve = f'''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

class Req(BaseModel):
{chr(10).join([f"    {k}: {('int' if v=='int' else 'float')}" for k,v in (request_schema or {}).items()])}

app = FastAPI()
model = joblib.load("{model_path}")
transformer = joblib.load("{transformer_path}")

@app.post("/predict")
def predict(payload: Req):
    df = pd.DataFrame([payload.dict()])
    X = transformer.transform(df)
    pred = model.predict(X)
    return {{"prediction": int(pred[0])}}
'''
        dockerfile = f'''
FROM python:3.10-slim
WORKDIR /app
COPY artifacts /app/artifacts
RUN pip install fastapi uvicorn joblib pandas scikit-learn
CMD ["uvicorn","artifacts.{run_id}_serve:app","--host","0.0.0.0","--port","8080"]
'''
    os.makedirs("artifacts", exist_ok=True)
    serve_path = f"artifacts/{run_id}_serve.py"
    docker_path = f"artifacts/{run_id}_Dockerfile"
    with open(serve_path, "w", encoding="utf-8") as f:
        f.write(serve)
    with open(docker_path, "w", encoding="utf-8") as f:
        f.write(dockerfile)
    return {"serve": serve_path, "dockerfile": docker_path}
