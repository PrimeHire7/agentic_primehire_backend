# reset_interview_attempts.py
import sys
from sqlalchemy import create_engine, text
from app.db import Base, engine  # uses your configured DB
from app.mcp.tools.interview_attempts_model import InterviewAttempts

def reset_table():
    print("\n====================================")
    print("🔥 RESETTING interview_attempts TABLE")
    print("====================================\n")

    try:
        # Create a connection
        conn = engine.connect()

        print("Dropping table if exists...")
        conn.execute(text("DROP TABLE IF EXISTS interview_attempts CASCADE;"))
        conn.commit()

        print("Recreating table using SQLAlchemy model...")
        Base.metadata.create_all(bind=engine)

        print("\n✅ interview_attempts recreated successfully!")
        print("Checking schema...\n")

        result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='interview_attempts';"))
        columns = result.fetchall()

        for col in columns:
            print(f"→ {col[0]} : {col[1]}")

        conn.close()
        print("\n🎉 DONE! Table is fresh and correct.\n")

    except Exception as e:
        print("\n❌ ERROR while resetting table:", e)
        sys.exit(1)

if __name__ == "__main__":
    reset_table()
