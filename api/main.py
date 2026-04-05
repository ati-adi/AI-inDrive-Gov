import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np

from scorer import build_score_card
from llm_explainer import get_explanation, chat_about_application
from database import init_db, save_decision, save_scoring_log, get_decision_stats, get_undecided_count
from models.xgboost_model import has_saved_model

app = FastAPI(title="SubsidyAI", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WEIGHTS = {}
DF = pd.DataFrame()


def find_application(app_number: str):
    if DF.empty:
        raise HTTPException(503, "Data not loaded")
    df_nums = DF["app_number"].astype(float).astype(np.int64).astype(str)
    try:
        search = str(int(float(app_number)))
    except ValueError:
        search = str(app_number)
    mask = df_nums == search
    if mask.sum() == 0:
        raise HTTPException(404, f"Application {app_number} not found")
    return DF[mask].iloc[0]


def load_data():
    global DF, WEIGHTS
    csv_path = DATA_DIR / "scored_applications.csv"
    meta_path = DATA_DIR / "pipeline_meta.json"
    if not csv_path.exists():
        return
    DF = pd.read_csv(csv_path)
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            WEIGHTS = meta.get("weights", {})
    print(f"Loaded {len(DF)} scored applications")


@app.on_event("startup")
async def startup():
    init_db()
    load_data()


FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FRONTEND_PATH.read_text(encoding="utf-8")
    return HTMLResponse("<h1>frontend not found</h1>", status_code=404)


class DecisionRequest(BaseModel):
    decision: str
    comment: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    history: list = []

class ScoreRequest(BaseModel):
    direction_code: str
    district: str
    oblast: str
    amount: float
    rate: float
    subsidy_name: Optional[str] = ""


def _build_isj(row):
    if "reliability_score" not in row.index:
        return None
    return {
        "years_active": int(row.get("years_active", 0)),
        "herd_trend": str(row.get("herd_trend", "unknown")),
        "reliability_score": float(row.get("reliability_score", 0)),
        "is_new_farmer": bool(row.get("is_new_farmer", False)),
        "herd_declining": bool(row.get("herd_declining", False)),
        "expected_mortality": float(row.get("expected_mortality", 0)),
        "actual_mortality": float(row.get("actual_mortality", 0)),
        "mortality_deviation": float(row.get("mortality_deviation", 0)),
        "pasture_norm_ha": float(row.get("pasture_norm_ha", 10)),
    }


def _build_violations(row):
    violations = []
    for col in [c for c in row.index if c.startswith("rule_") and "violations" not in c and "count" not in c]:
        if row.get(col, 1) == 0:
            violations.append(col.replace("rule_", "").replace("_", " "))
    return violations


@app.get("/applications")
async def list_applications(
    recommendation: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None),
    max_score: Optional[float] = Query(None),
    sort_by: str = Query("final_score"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    if DF.empty:
        raise HTTPException(503, "Data not loaded")
    filtered = DF.copy()
    if recommendation and "recommendation" in filtered.columns:
        filtered = filtered[filtered["recommendation"] == recommendation]
    if district:
        filtered = filtered[filtered["district"].str.contains(district, case=False, na=False)]
    if direction:
        filtered = filtered[filtered["direction_code"] == direction]
    if min_score is not None:
        filtered = filtered[filtered["final_score"] >= min_score]
    if max_score is not None:
        filtered = filtered[filtered["final_score"] <= max_score]
    ascending = sort_order == "asc"
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending)
    total = len(filtered)
    start = (page - 1) * per_page
    page_data = filtered.iloc[start:start + per_page]
    cols = ["app_number", "district", "direction_code", "amount", "unit_count",
            "final_score", "recommendation", "recommendation_ru"]
    available = [c for c in cols if c in page_data.columns]
    return {
        "total": total, "page": page, "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "items": page_data[available].replace({np.nan: None}).to_dict("records"),
    }


@app.get("/applications/{app_number}")
async def get_application(app_number: str):
    row = find_application(app_number)
    card = build_score_card(row, WEIGHTS)
    return {
        "score_card": card,
        "isj_data": _build_isj(row),
        "rule_violations": _build_violations(row),
        "raw_data": {
            "status": row.get("status", ""),
            "oblast": row.get("oblast", ""),
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "is_anomaly": bool(row.get("is_anomaly", False)),
            "coordination_flag": bool(row.get("coordination_flag", False)),
        },
    }


@app.get("/applications/{app_number}/explain")
async def explain_application(app_number: str):
    row = find_application(app_number)
    card = build_score_card(row, WEIGHTS)
    explanation = get_explanation(card, isj_data=_build_isj(row))
    save_scoring_log(
        app_number=app_number, score=card["final_score"],
        recommendation=card["recommendation"],
        components_json=json.dumps(card["components"], ensure_ascii=False),
        explanation=explanation,
    )
    return {"app_number": app_number, "score": card["final_score"],
            "recommendation": card["recommendation"], "explanation": explanation}


@app.post("/applications/{app_number}/chat")
async def chat_application(app_number: str, req: ChatRequest):
    row = find_application(app_number)
    card = build_score_card(row, WEIGHTS)
    response = chat_about_application(
        score_card=card, history=req.history, user_message=req.message,
        rule_violations=_build_violations(row), isj_data=_build_isj(row),
    )
    return {"app_number": app_number, "response": response}


@app.post("/applications/{app_number}/decision")
async def submit_decision(app_number: str, req: DecisionRequest):
    row = find_application(app_number)
    ai_score = float(row.get("final_score", 0))
    ai_rec = str(row.get("recommendation", ""))
    decision_id = save_decision(
        app_number=app_number, ai_score=ai_score, ai_recommendation=ai_rec,
        decision=req.decision, comment=req.comment or "",
    )
    return {
        "app_number": app_number, "decision": req.decision,
        "ai_recommendation": ai_rec, "matches_ai": (ai_rec == req.decision),
        "decision_id": decision_id, "status": "saved",
    }


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global DF, WEIGHTS
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Only .xlsx/.xls")
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / "upload.xlsx"
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    try:
        from pipeline import run_pipeline, save_results
        predict_only = has_saved_model()
        output = run_pipeline(mode="with_isj", predict_only=predict_only, verbose=True)
        save_results(output)
        load_data()
        return output["results"]
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    try:
        from pipeline import run_pipeline, save_results
        output = run_pipeline(mode="with_isj", predict_only=False, verbose=True)
        save_results(output)
        load_data()
        return output["results"]
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/stats")
async def get_stats():
    if DF.empty:
        raise HTTPException(503, "Data not loaded")
    rec_dist = DF["recommendation"].value_counts().to_dict() if "recommendation" in DF.columns else {}
    decision_stats = get_decision_stats()
    return {
        "total_applications": len(DF),
        "recommendation_distribution": {str(k): int(v) for k, v in rec_dist.items()},
        "score_stats": {
            "mean": round(float(DF["final_score"].mean()), 1),
            "median": round(float(DF["final_score"].median()), 1),
        },
        "decisions": decision_stats,
        "weights": WEIGHTS,
        "model_trained": has_saved_model(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
