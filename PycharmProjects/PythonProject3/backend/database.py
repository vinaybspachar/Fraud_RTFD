# backend/database.py

import snowflake.connector
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def load_data_from_snowflake(query: str) -> pd.DataFrame:
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    role = os.getenv("SNOWFLAKE_ROLE")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")

    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        role=role,
        warehouse=warehouse,
        database=database,
        schema=schema
    )

    # Run the provided query
    df = pd.read_sql(query, conn)
    conn.close()
    return df
