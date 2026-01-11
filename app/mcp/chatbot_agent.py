"""
Chatbot agent - conversational parsing and routing.

This function is registered as a tool named 'chatbot.handle_message' and
will call other tools via server_core.run_tool(...).
"""

import json
import logging
from app.mcp.server_core import register_mcp_tool, run_tool

logger = logging.getLogger(__name__)


def parse_jd_request(content: str):
    """
    Parse input like:
    [JD Creator] AI Engineer, 3 years, Bangalore, skills: Python, NLP
    """
    try:
        parts = [p.strip() for p in content.split(",")]
        role = parts[0] if parts else "Unknown Role"
        years = (
            int(parts[1].split()[0])
            if len(parts) > 1 and parts[1].split()[0].isdigit()
            else 0
        )
        location = parts[2] if len(parts) > 2 else "Remote"
        skills = []
        if len(parts) > 3 and "skills" in parts[3].lower():
            skills = [s.strip() for s in parts[3].split(":")[1].split(",")]
        return role, years, location, skills
    except Exception as e:
        logger.error("JD parse error: %s", e)
        return "Unknown Role", 0, "Remote", []


@register_mcp_tool(
    name="chatbot.handle_message",
    description="Handle simple chat commands and route to tools",
)
async def handle_message(message: str):
    logger.info("[CHATBOT] Incoming: %s", message)

    # 🔹 Zoho Bridge
    if message.startswith("[ZohoBridge]"):
        email = message.replace("[ZohoBridge]", "").strip()
        if not email:
            return "Please provide an email after [ZohoBridge]."
        result = await run_tool("zoho.fetch_candidates", {"email": email})
        return f"✅ Fetched {result.get('count', len(result.get('candidates', [])))} candidates."

    # 🔹 JD Creator
    if message.startswith("[JD Creator]"):
        content = message.replace("[JD Creator]", "").strip()
        role, years, location, skills = parse_jd_request(content)
        jd = await run_tool(
            "jd.generate",
            {"role": role, "years": years, "location": location, "skills": skills},
        )
        return f"📝 JD generated:\n{json.dumps(jd, indent=2)}"

    # 🔹 Resume Uploader (Instructional)
    if message.startswith("[Upload Resumes]"):
        return "📎 To upload resumes, use the Upload Resumes UI or call the /upload_resumes endpoint with files."

    # 🔹 Profile Matcher (trigger only on explicit user message)
    if message.startswith("[Profile Matcher]"):
        jd_text = message.replace("[Profile Matcher]", "").strip()
        if not jd_text:
            return "❌ Please paste a JD after [Profile Matcher]."
        logger.info("[CHATBOT] Running Profile Matcher for explicit request.")
        result = await run_tool("matcher.match_candidates", {"jd_text": jd_text})
        return result

    # 🔹 Interview Bot
    if message.startswith("[Interview Bot]"):
        query = message.replace("[Interview Bot]", "").strip()
        if not query:
            return "❌ Please provide candidate details after [Interview Bot]."
        result = await run_tool("interview_bot.run", {"query": query})
        return result

    # 🔸 Fallback
    return (
        "🤖 I didn't understand. Use one of the following commands:\n"
        "- [JD Creator] AI Engineer, 3 years, Bangalore, skills: Python, NLP\n"
        '- [Profile Matcher] {"role": "AI Engineer", "skills": ["Python"]}\n'
        "- [Upload Resumes]\n"
        "- [ZohoBridge] your@email.com\n"
        "- [Interview Bot] Candidate details"
    )
