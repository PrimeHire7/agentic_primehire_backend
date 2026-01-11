from app.db import Base, engine

# import all models so SQLAlchemy knows them
from app.mcp.tools.interview_attempts_model import InterviewAttempts
from app.mcp.tools.jd_history import GeneratedJDHistory
from app.mcp.tools.resume_tool import Candidate

def init_db():
    print("🟡 Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    print("🟢 Database schema ready.")
