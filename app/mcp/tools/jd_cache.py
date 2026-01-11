# # 📁 app/mcp/tools/jd_cache.py

# from fastapi import APIRouter, HTTPException
# from sqlalchemy import text
# from app.db import SessionLocal, engine
# from datetime import datetime
# import uuid

# router = APIRouter()

# # ============================================================
# # 1️⃣ CREATE TABLE IF NOT EXISTS
# # ============================================================

# CREATE_TABLE_SQL = """
# CREATE TABLE IF NOT EXISTS temp_jd_cache (
#     id SERIAL PRIMARY KEY,
#     token TEXT UNIQUE NOT NULL,
#     jd_text TEXT NOT NULL,
#     created_at TIMESTAMP DEFAULT NOW()
# );
# """

# # Run table creation ON IMPORT
# with engine.connect() as conn:
#     try:
#         conn.execute(text(CREATE_TABLE_SQL))
#         print("🆗 temp_jd_cache table ensured.")
#     except Exception as e:
#         print("❌ Failed to create temp_jd_cache table:", e)


# # ============================================================
# # 2️⃣ ROUTE → SAVE JD TEXT & RETURN TOKEN
# # ============================================================

# @router.post("/save")
# async def save_jd_text(payload: dict):
#     jd_text = payload.get("jd_text")

#     if not jd_text:
#         raise HTTPException(status_code=400, detail="jd_text is required")

#     # Create short unique token
#     token = "JD_" + uuid.uuid4().hex[:8]

#     session = SessionLocal()
#     try:
#         session.execute(
#             text(
#                 """
#                 INSERT INTO temp_jd_cache (token, jd_text)
#                 VALUES (:token, :text)
#                 """
#             ),
#             {"token": token, "text": jd_text},
#         )
#         session.commit()

#         return {"ok": True, "jd_token": token}

#     except Exception as e:
#         session.rollback()
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         session.close()


# # ============================================================
# # 3️⃣ ROUTE → FETCH JD TEXT USING TOKEN
# # ============================================================

# @router.get("/{token}")
# async def fetch_jd_text(token: str):
#     session = SessionLocal()
#     try:
#         row = session.execute(
#             text("SELECT jd_text FROM temp_jd_cache WHERE token = :token"),
#             {"token": token},
#         ).fetchone()

#         if not row:
#             raise HTTPException(status_code=404, detail="JD text not found")

#         return {"ok": True, "jd_text": row[0]}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         session.close()
# 📁 app/mcp/tools/jd_cache.py

from fastapi import APIRouter, HTTPException
from sqlalchemy import Column, Integer, Text, String, DateTime
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from app.db import Base, SessionLocal

router = APIRouter()


# ============================================================
# ORM MODEL
# ============================================================

class TempJDCache(Base):
    __tablename__ = "temp_jd_cache"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    jd_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================
# ROUTE → SAVE JD TEXT & RETURN TOKEN
# ============================================================

@router.post("/save")
async def save_jd_text(payload: dict):
    jd_text = payload.get("jd_text")

    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text is required")

    token = "JD_" + uuid.uuid4().hex[:8]

    session: Session = SessionLocal()
    try:
        entry = TempJDCache(
            token=token,
            jd_text=jd_text.strip(),
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)

        return {"ok": True, "jd_token": token}

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        session.close()


# ============================================================
# ROUTE → FETCH JD TEXT USING TOKEN
# ============================================================

@router.get("/{token}")
async def fetch_jd_text(token: str):
    session: Session = SessionLocal()
    try:
        entry = session.query(TempJDCache).filter(TempJDCache.token == token).first()

        if not entry:
            raise HTTPException(status_code=404, detail="JD text not found")

        return {"ok": True, "jd_text": entry.jd_text}

    finally:
        session.close()
