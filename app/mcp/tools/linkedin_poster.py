import os
import time
import json
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, RedirectResponse

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter()

LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")
TOKEN_FILE = "linkedin_tokens.json"

# -------------------- Pydantic Model --------------------
class LinkedInPostRequest(BaseModel):
    author_urn: str  # e.g., urn:li:person:YOUR_PERSON_URN
    text: str
    visibility: str = "PUBLIC"

# -------------------- Helper --------------------
def save_token(token_data: dict):
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f)

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return None

async def get_access_token():
    token_data = load_token()
    if token_data and token_data.get("expires_at", 0) > int(time.time()):
        return token_data["access_token"]
    raise HTTPException(status_code=401, detail="LinkedIn token expired or missing")

# -------------------- Generate OAuth URL --------------------
@router.get("/generate-token")
def generate_token_url():
    url = (
        "https://www.linkedin.com/oauth/v2/authorization?"
        f"response_type=code&client_id={LINKEDIN_CLIENT_ID}"
        f"&redirect_uri={LINKEDIN_REDIRECT_URI}"
        f"&scope=w_member_social%20r_liteprofile%20r_emailaddress"
    )
    return {"url": url}

# -------------------- OAuth Callback --------------------
@router.get("/callback")
async def linkedin_callback(code: str = Query(...)):
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data)
        if resp.status_code != 200:
            logger.error(f"[LinkedIn] Token generation failed: {resp.text}")
            raise HTTPException(status_code=400, detail=resp.text)
        token_data = resp.json()
        token_data["obtained_at"] = int(time.time())
        token_data["expires_at"] = int(time.time()) + int(token_data.get("expires_in", 5184000))
        save_token(token_data)
        return JSONResponse({"message": "✅ Token saved", "access_token": token_data["access_token"]})

# -------------------- Post JD --------------------
async def post_to_linkedin(author_urn: str, text: str, visibility: str = "PUBLIC"):
    access_token = await get_access_token()
    url = "https://api.linkedin.com/v2/ugcPosts"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": visibility
        }
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 201:
            logger.error(f"[LinkedIn POST] Failed: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"status": "success", "post_urn": resp.headers.get("X-RestLi-Id")}

@router.post("/post-text")
async def linkedin_post_text_endpoint(payload: LinkedInPostRequest):
    try:
        result = await post_to_linkedin(payload.author_urn, payload.text, payload.visibility)
        return {"ok": True, "result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
