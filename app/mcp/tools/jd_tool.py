# jd_engine_v3.py
import os
import re
import json
import math
import logging
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.mcp.server_core import register_mcp_tool
from app.db import SessionLocal
from app.mcp.tools.jd_history import GeneratedJDHistory

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# right after logger and setLevel
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY missing — JD Engine V3 won't function")

# --------------------
# Models
# --------------------
class JDRequest(BaseModel):
    role: str
    years: int
    location: str
    skills: list[str] = []
    job_type: str = "Full-time"
    company_name: str = ""
    about_company: str = ""
    responsibilities: list[str] = []
    qualifications: list[str] = []
    tone: str = "Corporate"  # Corporate | Startup | Friendly

class JDSinglePromptRequest(BaseModel):
    prompt: str
    tone: str = "Corporate"

# --------------------
# OpenAI helper
# --------------------
async def call_openai(messages, temperature=0.2, model="gpt-4o-mini", timeout_s=60):
    logger.debug(f"[OpenAI] Calling model={model} temperature={temperature}")
    logger.debug(f"[OpenAI] Messages: {messages!r}")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            logger.debug(f"[OpenAI] Status code: {resp.status_code}")
            text = resp.text or ""
            logger.debug(f"[OpenAI] Raw response (first 1000 chars): {text[:1000]}")
    except httpx.RequestError as re:
        logger.exception("[OpenAI] Network error calling OpenAI")
        raise Exception(f"OpenAI request error: {re}") from re

    if resp.status_code != 200:
        logger.error(f"[OpenAI] Non-200: {resp.status_code} body: {text[:1000]}")
        raise Exception(f"OpenAI returned {resp.status_code}: {text}")

    try:
        body = resp.json()
        # Safely extract content
        content = body["choices"][0]["message"]["content"]
        if content is None:
            raise KeyError("content is None")
        return content.strip()
    except Exception as e:
        logger.exception("[OpenAI] Failed to parse response JSON")
        raise Exception("Failed to parse OpenAI response") from e


# --------------------
# Utilities
# --------------------
def clean_html(text: str) -> str:
    # remove markdown fences and stray hashes
    if not text:
        return ""
    text = re.sub(r"```(?:html|json|md)?", "", text, flags=re.I)
    text = text.replace("```", "")
    text = text.replace("##", "").strip()
    return text

def clean_years(value):
    if not value:
        return 0
    s = str(value).lower()
    word_to_num = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
        "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12
    }
    if s in word_to_num:
        return word_to_num[s]
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0

def label_seniority(years:int) -> str:
    if years <= 1: return "Intern/Entry"
    if years <= 3: return "Junior"
    if years <= 6: return "Mid"
    if years <= 10: return "Senior"
    return "Staff/Principal"

def score_confidence_from_text(text: str) -> int:
    # A small heuristic score from text length & presence of bullets
    s = len(text)
    score = min(95, 40 + int(math.log(max(1,s))*6))
    if "-" in text or "\n" in text: score += 5
    return min(100, score)

# --------------------
# Smart helpers (LLM-assisted)
# --------------------
async def cluster_skills(skills:list[str], role:str) -> list[dict]:
    """
    Returns list of clusters with label and skills
    LLM helps grouping similar skills into clusters.
    """
    if not skills:
        return []
    prompt = f"""
You are a concise skill clustering assistant.
Input role: {role}
Skills: {', '.join(skills)}

Return JSON array of clusters. Each cluster is:
{{ "label": "<label>", "skills": ["s1","s2", ...] }}

Example:
[{{"label":"Programming","skills":["python","java"]}},{{"label":"Cloud","skills":["aws","gcp"]}}]
"""
    try:
        raw = await call_openai([{"role":"user","content":prompt}], temperature=0)
        raw = raw.replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        logger.warning("Skill clustering failed, falling back to heuristic")
        # fallback: basic buckets
        buckets = {"core":[], "infra":[], "data":[], "ml":[], "other":[]}
        for s in skills:
            ss = s.lower()
            if any(k in ss for k in ["python","java","c++","node","react","typescript"]): buckets["core"].append(s)
            elif any(k in ss for k in ["aws","gcp","azure","docker","kubernetes"]): buckets["infra"].append(s)
            elif any(k in ss for k in ["sql","postgres","mongodb","redis"]): buckets["data"].append(s)
            elif any(k in ss for k in ["pytorch","tensorflow","nlp","llm"]): buckets["ml"].append(s)
            else: buckets["other"].append(s)
        return [{"label":k,"skills":v} for k,v in buckets.items() if v]

async def suggest_responsibilities_and_qualifications(role:str, skills:list[str], years:int, tone:str):
    """
    If responsibilities/qualifications missing in the request,
    auto-suggest them using LLM.
    """
    prompt = f"""
You are a senior recruiter. Create 6 concise responsibilities and 5 preferred qualifications for the role.
Role: {role}
Years: {years}
Tone: {tone}
Skills: {', '.join(skills)}

Return JSON:
{{"responsibilities":["..."], "qualifications":["..."]}}
Keep lines short and action-oriented.
"""
    try:
        raw = await call_openai([{"role":"user","content":prompt}], temperature=0.2)
        raw = raw.replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        resp = parsed.get("responsibilities") or []
        qual = parsed.get("qualifications") or []
        return resp, qual
    except Exception as e:
        logger.warning("Auto-suggest failed, using defaults")
        default_resp = [
            "Design, implement and maintain core features.",
            "Collaborate with product and cross-functional teams.",
            "Ensure reliability, performance and scalability.",
            "Write tests and participate in code reviews.",
            "Mentor peers and contribute to technical decisions.",
            "Own delivery end-to-end."
        ]
        default_qual = [
            "Strong problem-solving skills.",
            "Experience with modern tooling and CI/CD.",
            "Demonstrated delivery on production systems.",
            "Good communication skills.",
            "Relevant domain experience."
        ]
        return default_resp, default_qual

async def semantic_validate_jd(html_jd:str, role:str, skills:list[str], years:int):
    """
    Ask LLM to give a short validation: score 0-100 and brief issues list
    """
    prompt = f"""
You are a quality checker. Evaluate the HTML job description below for:
- correctness for role: {role}
- coverage of skills: {', '.join(skills)}
- experience match: {years} years

Return JSON:
{{"score": <0-100>, "issues": ["..."]}}
HTML:
{html_jd}
"""
    try:
        raw = await call_openai([{"role":"user","content":prompt}], temperature=0)
        raw = raw.replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        score = int(parsed.get("score", score_confidence_from_text(html_jd)))
        issues = parsed.get("issues", [])
        return max(0,min(100,score)), issues
    except Exception as e:
        logger.warning("Semantic validation failed, using heuristic")
        logger.debug("[SEMANTIC] Validating generated HTML...")
        logger.debug(html_jd[:400])

        return score_confidence_from_text(html_jd), []

# --------------------
# Core JD formatter (HTML-only)
# --------------------
@register_mcp_tool(
    name="jd.generate.v3",
    description="JD Engine V3 - polished HTML output with metadata"
)
async def generate_jd_v3(
    role: str,
    years: int,
    location: str,
    skills: list = None,
    job_type: str = "Full-time",
    company_name: str = "",
    about_company: str = "",
    responsibilities: list = None,
    qualifications: list = None,
    tone: str = "Corporate"
):
    skills = skills or []
    responsibilities = responsibilities or []
    qualifications = qualifications or []

    # 1) Auto-suggest if missing
    if not responsibilities or not qualifications:
        auto_resp, auto_qual = await suggest_responsibilities_and_qualifications(role, skills, years, tone)
        if not responsibilities:
            responsibilities = auto_resp
        if not qualifications:
            qualifications = auto_qual

    # 2) Skill clustering
    skill_clusters = await cluster_skills(skills, role)

    # 3) Generate final HTML JD from LLM template
    skills_html = "\n".join([f"<li>{s}</li>" for s in skills]) or "<li>Not specified</li>"
    resp_html = "\n".join([f"<li>{r}</li>" for r in responsibilities])
    qual_html = "\n".join([f"<li>{q}</li>" for q in qualifications])

    html_prompt = f"""
You are PRIMEHIRE JD ENGINE V3.
Output STRICT PURE HTML ONLY. No markdown, no code fences.

Tone: {tone}
Role: {role}
Years: {years}
Location: {location}
Company: {company_name}
About Company: {about_company}
Skills: {', '.join(skills)}

Produce a polished LinkedIn-style HTML job description. Use only <h2>, <p>, <ul>, <li>.
Keep paragraphs short. Use action verbs. Keep it ATS-friendly.

OUTPUT HTML:
<h2>Job Title</h2>
<p>{role}</p>

<h2>About the Company</h2>
<p>{about_company or "We are a fast-growing company looking for talent."}</p>

<h2>About the Role</h2>
<p>The {role} will work on product & platform initiatives, collaborating with cross-functional teams, and delivering high-quality, production-grade solutions.</p>

<h2>Key Responsibilities</h2>
<ul>
{resp_html}
</ul>

<h2>Required Skills</h2>
<ul>
{skills_html}
</ul>

<h2>Preferred Qualifications</h2>
<ul>
{qual_html}
</ul>

<h2>Location</h2>
<p>{location or "Remote / Hybrid"}</p>

<h2>Experience</h2>
<p>{years}+ years</p>

<h2>Job Type</h2>
<p>{job_type}</p>

<h2>How to Apply</h2>
<p>Please submit your resume if you are interested and meet the requirements.</p>
"""
    raw_html = await call_openai([{"role":"user","content":html_prompt}], temperature=0.2)
    cleaned_html = clean_html(raw_html)

    # --- WRAP THE GENERATED JD HTML WITH A COPY BUTTON FOR UX ---
    # We build the wrapper using plain strings to avoid f-string brace interpolation issues.
    html_jd = (
        '<div class="jd-wrapper" style="position:relative; padding-top:44px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.03); padding:16px; background:#fff;">'
        # Inline copy button — copies the visible textual content of the .jd-content element.
        '<button '
        'onclick="(function(btn){'
        "try{"
        "  var text = btn.parentElement.querySelector('.jd-content').innerText || '';"
        "  navigator.clipboard.writeText(text).then(function(){"
        "    btn.textContent='✔ Copied';"
        "    setTimeout(function(){ btn.textContent='📋 Copy JD'; }, 2000);"
        "  }).catch(function(){ btn.textContent='❌ Failed'; setTimeout(function(){ btn.textContent='📋 Copy JD'; },2000); });"
        "}catch(e){ console.error('copy failed', e); btn.textContent='❌'; setTimeout(function(){ btn.textContent='📋 Copy JD'; },2000); }"
        "})(this)"
        '" '
        'aria-label="Copy job description" '
        'style="position:absolute; top:8px; right:8px; background:#f7f7f7; border:1px solid #ddd; border-radius:6px; padding:6px 10px; font-size:13px; cursor:pointer;">'
        '📋 Copy JD'
        '</button>'
        # JD content container
        f'<div class="jd-content" style="line-height:1.5; color:#1f2937;">{cleaned_html}</div>'
        '</div>'
    )


    # 4) Semantic validation scoring
    score, issues = await semantic_validate_jd(html_jd, role, skills, years)

    # 5) Seniority label
    seniority = label_seniority(years)

    # 6) Output metadata
    meta = {
        "skill_clusters": skill_clusters,
        "seniority": seniority,
        "score": score,
        "issues": issues,
        "suggested_responsibilities": responsibilities,
        "suggested_qualifications": qualifications,
        "tone": tone
    }

    return {
        "role": role,
        "html_jd": html_jd,
        "meta": meta
    }

# --------------------
# Endpoints (structured + single prompt)
# --------------------
@router.post("/generate", summary="Generate JD (structured) - V3")
async def generate_jd_endpoint(payload: JDRequest):
    try:
        result = await generate_jd_v3(**payload.dict())
        return {"ok": True, "result": result}
    except Exception as e:
        logger.exception("JD generation V3 failed")
        raise HTTPException(status_code=500, detail=str(e))

# Extraction helper for single prompt (V3)
async def extract_jd_from_prompt_v3(prompt: str) -> dict:
    logger.debug(f"[EXTRACTOR] Raw prompt: {prompt}")
    extraction_prompt = f"""
Extract structured JD JSON from the text. Return ONLY valid JSON.

Schema:
{{
  "role": "",
  "years": 0,
  "location": "",
  "skills": [],
  "job_type": "Full-time",
  "company_name": "",
  "about_company": ""
}}

PROMPT:
\"\"\"{prompt}\"\"\"
"""
    raw = await call_openai([{"role":"user","content":extraction_prompt}], temperature=0)
    raw = raw.replace("```json","").replace("```","").strip()
    logger.debug(f"[EXTRACTOR] Raw LLM output: {raw}")

    try:
        data = json.loads(raw)
        logger.debug(f"[EXTRACTOR] Parsed JSON: {json.dumps(data, indent=2)}")
    except Exception as e:
        logger.exception(f"[EXTRACTOR] JSON parse failed. Raw output: {raw!r}")
        raise HTTPException(status_code=500, detail="Extractor did not return valid JSON")

    return {
        "role": data.get("role",""),
        "years": clean_years(data.get("years")),
        "location": data.get("location",""),
        "skills": data.get("skills",[]),
        "job_type": data.get("job_type","Full-time"),
        "company_name": data.get("company_name",""),
        "about_company": data.get("about_company","")
    }

@router.post("/generate/single", summary="Generate JD (single natural prompt) - V3")
async def generate_jd_single(req: JDSinglePromptRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "Prompt cannot be empty"}

    logger.info("\n=========== JD SINGLE: START ===========")

    extracted = await extract_jd_from_prompt_v3(prompt)

    role = extracted.get("role")
    years = extracted.get("years")
    location = extracted.get("location")
    skills = extracted.get("skills", [])
    company = extracted.get("company_name")
    about_company = extracted.get("about_company")
    tone = req.tone or "Corporate"

    jd_result = await generate_jd_v3(
        role=role,
        years=years,
        location=location,
        skills=skills,
        job_type="Full-time",
        company_name=company,
        about_company=about_company,
        responsibilities=None,
        qualifications=None,
        tone=tone
    )

    html_jd = jd_result["html_jd"]

    # ---------------- Save JD ---------------------
    try:
        session = SessionLocal()
        jd_row = GeneratedJDHistory(
            designation=role,
            skills=", ".join(skills),
            jd_text=html_jd,
            matches_json={"profile_matches": []},
            matched_candidate_ids=[]
        )
        session.add(jd_row)
        session.commit()
        session.refresh(jd_row)
        jd_id = jd_row.id

    except Exception as e:
        session.rollback()
        return {"ok": False, "error": str(e)}

    finally:
        session.close()

    # ✨ IMPORTANT — tell frontend to show confirmation buttons
    return {
        "ok": True,
        "jd_id": jd_id,
        "jd_html": html_jd,
        "message": "JD generated successfully!",
        "ask_confirmation": True,  
    }
