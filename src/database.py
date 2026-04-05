import os
import numpy as np
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB = f"sqlite:///{DB_DIR}/subsidy.db"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, autoincrement=True)
    app_number = Column(String, index=True)
    district = Column(String)
    oblast = Column(String)
    direction_code = Column(String)
    subsidy_name = Column(Text)
    amount = Column(Float)
    rate = Column(Float)
    unit_count = Column(Float)
    submitted_at = Column(DateTime)
    final_score = Column(Float)
    recommendation = Column(String)
    comp_xgb = Column(Float)
    comp_anomaly = Column(Float)
    comp_rules = Column(Float)
    comp_regional = Column(Float)
    comp_volume = Column(Float)
    comp_reliability = Column(Float)
    reliability_score = Column(Float)
    herd_trend = Column(String)
    years_active = Column(Integer)
    is_new_farmer = Column(Boolean)
    herd_declining = Column(Boolean)
    expected_mortality = Column(Float)
    actual_mortality = Column(Float)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean)
    coordination_flag = Column(Boolean)
    rule_violation_count = Column(Integer)
    status = Column(String)
    user_decision = Column(String)
    decided_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    app_number = Column(String, index=True)
    ai_score = Column(Float)
    ai_recommendation = Column(String)
    decision = Column(String)
    matches_ai = Column(Boolean)
    comment = Column(Text)
    decided_by = Column(String, default="operator")
    decided_at = Column(DateTime, default=datetime.utcnow)


class ScoringLog(Base):
    __tablename__ = "scoring_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    app_number = Column(String, index=True)
    score = Column(Float)
    recommendation = Column(String)
    components_json = Column(Text)
    explanation = Column(Text)
    scored_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()


def save_applications_from_df(df):
    session = get_session()
    try:
        session.query(Application).delete()
        records = []
        for _, row in df.iterrows():
            def sf(val, default=0.0):
                try:
                    v = float(val)
                    return v if not np.isnan(v) else default
                except (TypeError, ValueError):
                    return default
            def si(val, default=0):
                try:
                    v = float(val)
                    return int(v) if not np.isnan(v) else default
                except (TypeError, ValueError):
                    return default
            def sb(val):
                try:
                    return bool(val)
                except (TypeError, ValueError):
                    return False
            app = Application(
                app_number=str(row.get("app_number", "")),
                district=str(row.get("district", "")),
                oblast=str(row.get("oblast", "")),
                direction_code=str(row.get("direction_code", "")),
                subsidy_name=str(row.get("subsidy_name", "")),
                amount=sf(row.get("amount")),
                rate=sf(row.get("rate")),
                unit_count=sf(row.get("unit_count")),
                final_score=sf(row.get("final_score")),
                recommendation=str(row.get("recommendation", "")),
                comp_xgb=sf(row.get("comp_xgb")),
                comp_anomaly=sf(row.get("comp_anomaly")),
                comp_rules=sf(row.get("comp_rules")),
                comp_regional=sf(row.get("comp_regional")),
                comp_volume=sf(row.get("comp_volume")),
                comp_reliability=sf(row.get("comp_reliability")),
                reliability_score=sf(row.get("reliability_score")),
                herd_trend=str(row.get("herd_trend", "")),
                years_active=si(row.get("years_active")),
                is_new_farmer=sb(row.get("is_new_farmer")),
                herd_declining=sb(row.get("herd_declining")),
                expected_mortality=sf(row.get("expected_mortality")),
                actual_mortality=sf(row.get("actual_mortality")),
                anomaly_score=sf(row.get("anomaly_score")),
                is_anomaly=sb(row.get("is_anomaly")),
                coordination_flag=sb(row.get("coordination_flag")),
                rule_violation_count=si(row.get("rule_violation_count")),
                status=str(row.get("status", "")),
                user_decision=None,
                decided_at=None,
            )
            records.append(app)
        session.bulk_save_objects(records)
        session.commit()
        return len(records)
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def save_decision(app_number, ai_score, ai_recommendation, decision, comment="", decided_by="operator"):
    session = get_session()
    try:
        d = Decision(
            app_number=str(app_number),
            ai_score=ai_score,
            ai_recommendation=ai_recommendation,
            decision=decision,
            matches_ai=(ai_recommendation == decision),
            comment=comment,
            decided_by=decided_by,
        )
        session.add(d)
        app = session.query(Application).filter(Application.app_number == str(app_number)).first()
        if app:
            app.user_decision = decision
            app.decided_at = datetime.utcnow()
        session.commit()
        return d.id
    finally:
        session.close()


def save_scoring_log(app_number, score, recommendation, components_json, explanation=""):
    session = get_session()
    try:
        log = ScoringLog(
            app_number=str(app_number),
            score=score,
            recommendation=recommendation,
            components_json=components_json,
            explanation=explanation,
        )
        session.add(log)
        session.commit()
        return log.id
    finally:
        session.close()


def get_decisions_as_labels():
    session = get_session()
    try:
        decisions = session.query(Decision).all()
        labels = {}
        for d in decisions:
            labels[d.app_number] = 1 if d.decision == "approve" else 0
        return labels
    finally:
        session.close()


def get_decision_stats():
    session = get_session()
    try:
        total = session.query(Decision).count()
        approved = session.query(Decision).filter(Decision.decision == "approve").count()
        rejected = session.query(Decision).filter(Decision.decision == "reject").count()
        matches = session.query(Decision).filter(Decision.matches_ai == True).count()
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "ai_agreement_rate": round(matches / max(total, 1), 3),
        }
    finally:
        session.close()


def get_undecided_count():
    session = get_session()
    try:
        return session.query(Application).filter(Application.user_decision == None).count()
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    print(f"Database: {DATABASE_URL}")
