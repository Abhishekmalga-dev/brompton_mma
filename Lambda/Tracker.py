import os
import io
import boto3
import json
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import pandas as pd
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb")
s3_client = boto3.client("s3")

# ----------------------------------------------------------------
# Action router — keeps the existing availability-check behavior as
# the default (so the Step Function needs zero changes), and adds
# the S3 -> RDS load as a separate, independent code path selected
# via event["action"]. The two paths never run in the same
# invocation, so a failure in one cannot affect the other.
# ----------------------------------------------------------------
def lambda_handler(event, context):
    action = event.get("action", "check_availability")

    if action == "check_availability":
        return check_availability()
    elif action == "load_s3_to_rds":
        return load_s3_to_rds()
    else:
        raise ValueError(f"Unknown action: {action}")


# ----------------------------------------------------------------
# Existing behavior — unchanged from before
# ----------------------------------------------------------------
def check_availability():
    table_name = os.environ.get("TRACKER_TABLE_NAME")
    if not table_name:
        raise EnvironmentError("TRACKER_TABLE_NAME environment variable is not set")
    table = dynamodb.Table(table_name)

    # Compute today's date the same way the Curation Glue job does,
    # so the lookup key matches exactly what was written.
    execution_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"DEBUG: Looking up availability for execution_date={execution_date} in table={table_name}")

    try:
        response = table.get_item(Key={"execution_date": execution_date})
    except Exception as e:
        print(f"ERROR: Failed to read from DynamoDB table {table_name}: {e}")
        raise  # let the Step Function's Catch block handle this as a real failure

    item = response.get("Item")

    if not item:
        print(f"INFO: No availability record found for execution_date={execution_date}. Treating all sources as unavailable.")
        return {
            "ivr_available": False,
            "sms_web_relational_available": False,
            "sms_web_transactional_available": False,
            "api_web_transactional_available": False,
            "api_web_relational_available": False,
            "execution_date": execution_date,
            "lookup_status": "NOT_FOUND"
        }

    print(f"INFO: Found availability record for execution_date={execution_date}: {item}")

    return {
        "ivr_available": bool(item.get("ivr_available", False)),
        "sms_web_relational_available": bool(item.get("sms_web_relational_available", False)),
        "sms_web_transactional_available": bool(item.get("sms_web_transactional_available", False)),
        "api_web_transactional_available": bool(item.get("api_web_transactional_available", False)),
        "api_web_relational_available": bool(item.get("api_web_relational_available", False)),
        "execution_date": execution_date,
        "lookup_status": "FOUND"
    }


# ----------------------------------------------------------------
# RDS connection — reused verbatim from datalake-comprehend-corrected-output.py
# so both Lambdas share one operational pattern for RDS credentials.
# ----------------------------------------------------------------
def get_rds_connection():
    required_vars = [
        "RDS_HOST",
        "RDS_PORT",
        "RDS_DB_NAME",
        "RDS_USER",
        "RDS_PASSWORD",
    ]

    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise ValueError(f"Missing required RDS environment variables: {missing}")

    return psycopg2.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", "5432")),
        dbname=os.environ["RDS_DB_NAME"],
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        connect_timeout=10,
        sslmode=os.environ.get("RDS_SSLMODE", "prefer"),
    )


# ----------------------------------------------------------------
# S3 -> RDS load
#
# Source layout (fixed business structure, not environment config,
# so intentionally hardcoded rather than driven by env vars):
#   s3://{CURATED_BUCKET_NAME}/sentiment_analysis/aggregate/*.parquet     -> Postgres schema "aggregate"
#   s3://{CURATED_BUCKET_NAME}/sentiment_analysis/monthly_yago/*.parquet -> Postgres schema "monthly_yago"
#
# Each file becomes a table named after its filename (minus extension),
# e.g. api_relational.parquet -> aggregate.api_relational
#
# Strategy: full refresh (TRUNCATE + INSERT) each run, per business decision.
# Table columns are inferred automatically from each file's Parquet/pandas
# dtypes, since these are small reference-style files and manually
# maintaining 10 column lists would be more error-prone than deriving them.
# ----------------------------------------------------------------
PREFIX_TO_SCHEMA = {
    "sentiment_analysis/aggregate/": "aggregate",
    "sentiment_analysis/monthly_yago/": "monthly_yago",
}

SOURCE_FILES = [
    "api_relational.parquet",
    "api_transactional.parquet",
    "ivr.parquet",
    "sms_relational.parquet",
    "sms_transactional.parquet",
]


def _pg_type_for_dtype(dtype) -> str:
    """Map a pandas dtype to a Postgres column type."""
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    # Default: strings, objects, categoricals, anything else -> TEXT.
    # TEXT is a safe fallback in Postgres (no length limit, no truncation risk).
    return "TEXT"


def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    buffer = io.BytesIO(obj["Body"].read())
    return pd.read_parquet(buffer)


def create_table_if_not_exists(conn, schema_name: str, table_name: str, df: pd.DataFrame):
    columns_sql = [
        sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(_pg_type_for_dtype(dtype)))
        for col, dtype in df.dtypes.items()
    ]
    create_stmt = sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
        sql.SQL(", ").join(columns_sql),
    )
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
        )
        cur.execute(create_stmt)


def truncate_and_load(conn, schema_name: str, table_name: str, df: pd.DataFrame):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE TABLE {}.{}").format(
                sql.Identifier(schema_name), sql.Identifier(table_name)
            )
        )

        if df.empty:
            print(f"INFO: {schema_name}.{table_name} source file has 0 rows after truncate; nothing to insert.")
            return

        # NOTE: df.where(pd.notnull(df), None) is NOT reliable here -- with pandas'
        # newer native "str" dtype, .where() silently keeps NaN instead of writing
        # None, which would insert the literal text "NaN" into the DB instead of
        # a real SQL NULL. Converting per-value with pd.isna() at row-build time
        # avoids that dtype-dependent behavior entirely.
        rows = [
            tuple(None if pd.isna(value) else value for value in row)
            for row in df.itertuples(index=False, name=None)
        ]

        insert_stmt = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
            sql.Identifier(schema_name),
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in df.columns),
        )
        execute_values(cur, insert_stmt, rows)


def load_s3_to_rds():
    bucket = os.environ.get("CURATED_BUCKET_NAME")
    if not bucket:
        raise EnvironmentError("CURATED_BUCKET_NAME environment variable is not set")

    conn = get_rds_connection()
    results = []

    try:
        for prefix, schema_name in PREFIX_TO_SCHEMA.items():
            for filename in SOURCE_FILES:
                key = f"{prefix}{filename}"
                table_name = filename.replace(".parquet", "")

                print(f"DEBUG: Loading s3://{bucket}/{key} -> {schema_name}.{table_name}")

                try:
                    df = read_parquet_from_s3(bucket, key)
                    create_table_if_not_exists(conn, schema_name, table_name, df)
                    truncate_and_load(conn, schema_name, table_name, df)
                    conn.commit()
                    print(f"INFO: Loaded {len(df)} rows into {schema_name}.{table_name}")
                    results.append({
                        "schema": schema_name,
                        "table": table_name,
                        "rows_loaded": len(df),
                        "status": "SUCCESS"
                    })
                except Exception as e:
                    conn.rollback()
                    print(f"ERROR: Failed to load s3://{bucket}/{key} into {schema_name}.{table_name}: {e}")
                    results.append({
                        "schema": schema_name,
                        "table": table_name,
                        "rows_loaded": 0,
                        "status": "FAILED",
                        "error": str(e)
                    })
    finally:
        conn.close()

    failures = [r for r in results if r["status"] == "FAILED"]
    return {
        "lookup_status": "FAILED" if failures else "SUCCESS",
        "results": results
    }