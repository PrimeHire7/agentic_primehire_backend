# inspect_jd_table.py
import json
from sqlalchemy import inspect, text
from app.db import SessionLocal, engine

def inspect_table():
    inspector = inspect(engine)
    
    print("\n============================")
    print("📌 TABLE: generated_jd_history")
    print("============================")

    columns = inspector.get_columns("generated_jd_history")
    for col in columns:
        print(f"- {col['name']} | {col['type']} | nullable={col['nullable']}")
    
    print("\nFetching first 5 rows...\n")

    session = SessionLocal()
    try:
        query = text("SELECT id, designation, skills, jd_text, matches_json, created_at FROM generated_jd_history LIMIT 5")
        rows = session.execute(query).fetchall()
        
        for r in rows:
            print("----------------------------------------------------")
            print(f"ID: {r.id}")
            print(f"Designation: {r.designation}")
            print(f"Skills: {r.skills}")
            print(f"JD Text (preview): {str(r.jd_text)[:150]}...")
            print("Matches JSON:")
            
            try:
                pretty = json.dumps(r.matches_json, indent=2)
                print(pretty)
            except Exception:
                print("⚠️ matches_json is not valid JSON, raw value:")
                print(r.matches_json)

            print(f"Created At: {r.created_at}")
            print("----------------------------------------------------\n")
    
    finally:
        session.close()


if __name__ == "__main__":
    inspect_table()
