# app/mcp/tools/primehire_brain.py
import os
import json
import logging
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from datetime import datetime
from dotenv import load_dotenv
from app.mcp.server_core import register_mcp_tool
from .resume_tool import Candidate, Base  # 👈 import your existing Candidate model
from sqlalchemy import func
from fastapi import Query
from sqlalchemy import or_


# -------------------- ENV --------------------
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://primehire_user_naresh:primehire_user_naresh@localhost/primehireBrain_db",
)

engine = create_engine(DB_URL, echo=False)
Base.metadata.create_all(bind=engine)

router = APIRouter()


# -------------------- Tool: View Stored Candidates --------------------
@register_mcp_tool(
    name="resume.list_candidates",
    description="Fetch all stored candidates metadata from primehireBrain_db",
)
async def list_candidates_tool():
    """Used for MCP or AI call"""
    return await list_candidates()


# -------------------- API Endpoint --------------------
# @router.get("/list")
# async def list_candidates():
#     """Fetch all stored candidate metadata (unique by email, latest first)"""
#     try:
#         session = Session(bind=engine)
#         # Fetch only the latest entry for each email
#         subquery = (
#             session.query(
#                 Candidate.email, func.max(Candidate.last_updated).label("latest_update")
#             )
#             .group_by(Candidate.email)
#             .subquery()
#         )

#         candidates = (
#             session.query(Candidate)
#             .join(
#                 subquery,
#                 (Candidate.email == subquery.c.email)
#                 & (Candidate.last_updated == subquery.c.latest_update),
#             )
#             .order_by(Candidate.last_updated.desc())
#             .all()
#         )

#         results = [
#             {
#                 "candidate_id": c.candidate_id,
#                 "full_name": c.full_name,
#                 "email": c.email,
#                 "phone": c.phone,
#                 "linkedin_url": c.linkedin_url,
#                 "current_title": c.current_title,
#                 "current_company": c.current_company,
#                 "years_of_experience": c.years_of_experience,
#                 "top_skills": c.top_skills,
#                 "education_summary": c.education_summary,
#                 "location": c.location,
#                 "resume_link": c.resume_link,
#                 "rating_score": c.rating_score,
#                 "last_updated": (
#                     c.last_updated.strftime("%Y-%m-%d %H:%M:%S")
#                     if c.last_updated
#                     else None
#                 ),
#             }
#             for c in candidates
#         ]
#         logger.info(f"[list_candidates] ✅ Returned {len(results)} unique candidates")
#         return {"resumes": results}

#     except Exception as e:
#         logger.error(f"[list_candidates] ❌ Failed: {e}")
#         raise HTTPException(status_code=500, detail="Failed to fetch candidates")

#     finally:
#         session.close()

@router.get("/list")
async def list_candidates(search: str = Query(None)):
    """
    Fetch all stored candidate metadata
    - Unique by email
    - Latest first
    - Optional search by name or email
    """
    try:
        session = Session(bind=engine)

        subquery = (
            session.query(
                Candidate.email,
                func.max(Candidate.last_updated).label("latest_update"),
            )
            .group_by(Candidate.email)
            .subquery()
        )

        query = (
            session.query(Candidate)
            .join(
                subquery,
                (Candidate.email == subquery.c.email)
                & (Candidate.last_updated == subquery.c.latest_update),
            )
        )

        # 🔍 SEARCH FILTER
        if search:
            like = f"%{search.lower()}%"
            query = query.filter(
                or_(
                    func.lower(Candidate.full_name).like(like),
                    func.lower(Candidate.email).like(like),
                )
            )

        candidates = query.order_by(Candidate.last_updated.desc()).all()

        results = [
            {
                "candidate_id": c.candidate_id,
                "full_name": c.full_name,
                "email": c.email,
                "phone": c.phone,
                "location": c.location,
                "current_title": c.current_title,
                "current_company": c.current_company,
                "years_of_experience": c.years_of_experience,
                "top_skills": c.top_skills,
                "last_updated": (
                    c.last_updated.strftime("%Y-%m-%d %H:%M:%S")
                    if c.last_updated
                    else None
                ),
            }
            for c in candidates
        ]

        return {"resumes": results}

    except Exception as e:
        logger.error(f"[list_candidates] ❌ {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch candidates")

    finally:
        session.close()
# -------------------- Tool: Delete Candidate --------------------
@router.delete("/delete/{candidate_id}")
async def delete_candidate(candidate_id: str):
    """
    Delete a candidate from PostgreSQL and Pinecone.
    """
    try:
        session = Session(bind=engine)

        # 1️⃣ Find candidate by candidate_id
        candidate = (
            session.query(Candidate)
            .filter(Candidate.candidate_id == candidate_id)
            .first()
        )

        if not candidate:
            session.close()
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Store full_name for UI logs
        name = candidate.full_name

        # 2️⃣ Delete from PostgreSQL
        session.delete(candidate)
        session.commit()
        session.close()

        # 3️⃣ Delete from Pinecone
        try:
            namespace = "__default__"
            index.delete(ids=[candidate_id], namespace=namespace)
        except Exception as e:
            print(f"[Pinecone] ⚠️ Deletion failed but DB deleted: {e}")

        return {
            "status": "success",
            "message": f"Candidate '{name}' deleted successfully",
            "candidate_id": candidate_id
        }

    except Exception as e:
        print(f"[delete_candidate] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


