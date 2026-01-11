"""
Zoho tool wrapper

This implementation uses FastAPI endpoints on your FastAPI Zoho integration.
You already have a FastAPI endpoint like /fetch_candidates — this tool calls it.
"""
import os
import requests
from app.mcp.server_core import register_mcp_tool
from dotenv import load_dotenv
load_dotenv()

ZOHO_API_BASE = os.getenv("ZOHO_API_BASE", "https://primehire.nirmataneurotech.com")

@register_mcp_tool(name="zoho.fetch_candidates", description="Fetch candidate profiles from Zoho Recruit")
def fetch_candidates(email: str, page: int = 1, per_page: int = 50):
    url = f"{ZOHO_API_BASE}/fetch_candidates?email={email}&page={page}&per_page={per_page}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()
