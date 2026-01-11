import asyncio
import logging
from fastapi import APIRouter
from playwright.async_api import async_playwright
from app.mcp.server_core import register_mcp_tool

logger = logging.getLogger(__name__)
router = APIRouter()


# 🧰 Step 1: Open LinkedIn login page in Chromium
@register_mcp_tool(
    name="linkedin.open_login", description="Open a LinkedIn browser for manual login"
)
async def linkedin_open_login(_params=None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.linkedin.com/login")
        logger.info("Opened LinkedIn login page")
        # Wait for user to log in (up to 2 minutes)
        await page.wait_for_url("**/feed", timeout=120000)
        return {"status": "Logged in successfully!"}


# 🧰 Step 2: Automate job posting (example)
@register_mcp_tool(
    name="linkedin.post_job", description="Post a job on LinkedIn after login"
)
async def linkedin_post_job(params: dict):
    """
    Expected params:
    {
        "title": "AI Engineer",
        "description": "We are hiring...",
        "location": "Hyderabad, India",
        "company": "Nirmata Neurotech"
    }
    """
    title = params.get("title")
    desc = params.get("description")
    location = params.get("location", "Remote")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # Assume user is already logged in (persist context is better for prod)
        await page.goto("https://www.linkedin.com/talent/post-a-job")
        await page.wait_for_load_state("networkidle")

        # Fill in job title
        await page.fill('input[name="jobTitle"]', title)
        # Fill in location
        await page.fill('input[name="jobLocation"]', location)
        # Fill in description (selector may need adjustment)
        await page.fill("textarea", desc)

        # Submit (selector may differ depending on LinkedIn updates)
        await page.click('button[type="submit"]')

        await page.wait_for_timeout(5000)
        return {"status": f"Job '{title}' posted successfully!"}
