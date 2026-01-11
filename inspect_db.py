# inspect_db.py
import pandas as pd
from sqlalchemy import create_engine, inspect
from app.db import DATABASE_URL

def inspect_database():
    print("\n==============================")
    print("📊 DATABASE INSPECTION TOOL")
    print("==============================\n")

    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)

    # List all tables
    tables = inspector.get_table_names()
    print("📌 AVAILABLE TABLES:")
    for t in tables:
        print(" -", t)

    print("\n-----------------------------------------")

    # Preview each table
    for table in tables:
        print(f"\n📍 TABLE: {table}")
        # Show columns
        columns = inspector.get_columns(table)
        for col in columns:
            print(f"  - {col['name']} ({col['type']})")

        # Sample rows
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", engine)
            if df.empty:
                print("  ⚠ No records in this table.")
            else:
                print("\n  🔹 Example rows:")
                print(df.to_string(index=False))
        except Exception as e:
            print("  ❌ Could not fetch rows:", e)

        print("\n-----------------------------------------")


if __name__ == "__main__":
    inspect_database()
