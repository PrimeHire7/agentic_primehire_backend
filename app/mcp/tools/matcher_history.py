# 📁 app/mcp/tools/matcher_history.py
import os
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from sqlalchemy import Column, Integer, Text, JSON, TIMESTAMP, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter()

# ======================================================
# 🧠 DATABASE CONFIG
# ======================================================
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://primehire_user_naresh:primehire_user_naresh@localhost/primehireBrain_db",
)
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ======================================================
# 🧱 MODEL DEFINITION
# ======================================================
class MatcherHistory(Base):
    __tablename__ = "matcher_history"  # ✅ matches your real DB table

    id = Column(Integer, primary_key=True, autoincrement=True)
    jd_text = Column(Text, nullable=False)
    jd_meta = Column(JSON)
    candidates = Column(JSON)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

# Create table if not exists
Base.metadata.create_all(bind=engine)

# ======================================================
# 💾 SAVE FUNCTION
# ======================================================
def save_match_to_db(jd_text, jd_meta, candidates):
    """Store JD and candidate match data in matcher_history table."""
    session = SessionLocal()
    try:
        entry = MatcherHistory(
            jd_text=jd_text,
            jd_meta=jd_meta,
            candidates=candidates,
        )
        session.add(entry)
        session.commit()
        logger.info(f"💾 Saved match (JD: {jd_meta.get('role', 'N/A')}) with {len(candidates)} candidates.")
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Failed to save profile match history: {e}")
        raise HTTPException(status_code=500, detail="Failed to save match history.")
    finally:
        session.close()

# ======================================================
# 📜 ROUTES
# ======================================================
@router.get("/profile/history")
async def get_profile_match_history():
    """Fetch all profile match histories (latest first)."""
    try:
        session = SessionLocal()
        records = (
            session.query(MatcherHistory)
            .order_by(MatcherHistory.created_at.desc())
            .all()
        )
        history = [
            {
                "id": r.id,
                "jd_text": (r.jd_text[:200] + "...") if len(r.jd_text) > 200 else r.jd_text,
                "jd_meta": r.jd_meta,
                "total_candidates": len(r.candidates or []),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
        return {"history": history}
    except Exception as e:
        logger.exception("❌ Error fetching match history")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@router.get("/profile/history/{match_id}")
async def get_profile_match_detail(match_id: int):
    """Fetch details for a specific match record."""
    session = SessionLocal()
    try:
        record = session.query(MatcherHistory).filter_by(id=match_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Match record not found")

        return {
            "id": record.id,
            "jd_text": record.jd_text,
            "jd_meta": record.jd_meta,
            "candidates": record.candidates,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }
    except Exception as e:
        logger.exception("❌ Error fetching match detail")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()
