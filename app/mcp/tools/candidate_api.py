# app/mcp/tools/candidate_api.py

import json
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from app.mcp.tools.resume_tool import Candidate, engine  # ✅ correct import
from sqlalchemy.orm import sessionmaker

router = APIRouter()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ---------------------------------------------------------
# GET ALL CANDIDATES
# ---------------------------------------------------------
@router.get("/candidates/all")
async def get_all_candidates():
    session = SessionLocal()
    try:
        candidates = session.query(Candidate).order_by(Candidate.last_updated.desc()).all()

        results = []
        for c in candidates:
            results.append({
                "candidate_id": c.candidate_id,
                "full_name": c.full_name,
                "email": c.email,
                "phone": c.phone,
                "current_title": c.current_title,
                "current_company": c.current_company,
                "years_of_experience": c.years_of_experience,
                "location": c.location,
                "top_skills": c.top_skills,
                "resume_link": c.resume_link,
                "last_updated": c.last_updated.isoformat() if c.last_updated else None
            })

        return {"count": len(results), "candidates": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# ---------------------------------------------------------
# GET CANDIDATE BY EMAIL
# ---------------------------------------------------------
@router.get("/candidates/by_email/{email}")
async def get_candidate_by_email(email: str):
    session = SessionLocal()
    try:
        c = session.query(Candidate).filter(Candidate.email == email).first()

        if not c:
            raise HTTPException(status_code=404, detail="Candidate not found")

        return {
            "candidate_id": c.candidate_id,
            "full_name": c.full_name,
            "email": c.email,
            "phone": c.phone,
            "current_title": c.current_title,
            "current_company": c.current_company,
            "years_of_experience": c.years_of_experience,
            "location": c.location,
            "top_skills": c.top_skills.split(",") if c.top_skills else [],
            "resume_link": c.resume_link,
            "last_updated": c.last_updated.isoformat() if c.last_updated else None
        }
    finally:
        session.close()
