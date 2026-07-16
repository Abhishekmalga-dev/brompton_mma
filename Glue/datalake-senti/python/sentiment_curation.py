import sys
import boto3, json
from typing import List

from pyspark.sql.functions import broadcast, trim, lower
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number
from pyspark.sql.types import DoubleType, DateType

# INIT

required_args = ["JOB_NAME"]
optional_args = []
for opt in ["mode", "ENV", "ARREARS_JOIN_KEY", "CAS_JOIN_KEY", "STAGING"]:
    if f"--{opt}" in sys.argv:
        optional_args.append(opt)

args = getResolvedOptions(sys.argv, required_args + optional_args)
processing_mode = args.get("mode", "manual").lower()
ENV = args.get("ENV", "dev").lower()
ARREARS_JOIN_KEY = args.get("ARREARS_JOIN_KEY", "account_number")
CAS_JOIN_KEY = args.get("CAS_JOIN_KEY", "account_number")

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
print(f"Processing mode received: {processing_mode}")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3 = boto3.client("s3")

# PATHS
ACCOUNT_TIER = "nonprod" if ENV == "dev" else "prod"
TEMP_BUCKET = f"psegli-datalake{ACCOUNT_TIER}li-datalake-temp-{ENV}"
CURATED_BUCKET = f"psegli-datalake{ACCOUNT_TIER}li-datalake-curated-{ENV}"

# STAGING controls an optional path segment inserted into every S3 path below,
# so this same job can write to a parallel "staging" area for testing without
# touching real dev output. "dev" -> no segment (empty string). Any other
# value (e.g. "staging") is used as-is as the segment name.
STAGING_PARAM = args.get("STAGING", "dev").lower()
STAGING_SEGMENT = "" if STAGING_PARAM == "dev" else STAGING_PARAM
print(f"STAGING parameter received: '{STAGING_PARAM}' -> path segment: '{STAGING_SEGMENT or '(none)'}'")

def build_path(bucket: str, *parts) -> str:
    """
    Builds an s3:// path from a bucket and any number of path segments,
    silently skipping empty segments (like an empty STAGING_SEGMENT) so we
    never end up with a double slash. Always returns a trailing slash.
    e.g. build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/ivr")
    -> "s3://bucket/sentiment_analysis/final/ivr/"           (STAGING_SEGMENT="")
    -> "s3://bucket/staging/sentiment_analysis/final/ivr/"   (STAGING_SEGMENT="staging")
    """
    clean_parts = [p.strip("/") for p in parts if p]
    return f"s3://{bucket}/" + "/".join(clean_parts) + "/"

def build_prefix(*parts) -> str:
    """Same segment-skipping logic as build_path, but for a bare S3 prefix
    (no bucket, no s3:// scheme) — used where bucket and prefix are passed
    separately, e.g. get_latest_partition_value(bucket, prefix, ...)."""
    clean_parts = [p.strip("/") for p in parts if p]
    return "/".join(clean_parts) + "/"

CAS_BUCKET = CURATED_BUCKET
CAS_PREFIX = build_prefix(STAGING_SEGMENT, "cas")

ARREARS_S3_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "cas_arrears")

EVENT_DATE_PATH = build_path(TEMP_BUCKET, STAGING_SEGMENT, "ccaas/event_dates/survey_api_json")

FINAL_IVR_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/ivr")
FINAL_TXN_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/transactional")
FINAL_REL_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/relational")

CAS_JOIN_IVR_STAGING = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/cas_join/staging/ivr")
CAS_JOIN_TXN_STAGING = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/cas_join/staging/txn")
CAS_JOIN_REL_STAGING = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/cas_join/staging/rel")

CAS_JOIN_IVR_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/cas_join/ivr")
CAS_JOIN_TXN_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/cas_join/transactional")
CAS_JOIN_REL_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/cas_join/relational")

# HELPERS

def safe_read_parquet(path, label):
    clean_path = path.rstrip("/")
    try:
        print(f"checking input path for {label}: {clean_path}")
        df = spark.read.parquet(clean_path)
        count = df.count()
        print(f"{label} file found, count: {count}")
        return df
    except Exception as e:
        print(f"{label} file not available, skipping {label}: {str(e)}")
        return None

def enrich_with_reference(df, ref_df, ref_join_key):
    """
    Left-joins df with reference data. Survey-side key is customer_account_number
    (preferred) or cas_account_number (fallback), whichever is present in df.
    Reference-side key is ref_join_key, so it is always renamed to match the
    survey-side key before joining.
    Returns (enriched_df, join_key_used). join_key_used is None if neither
    survey-side key existed, in which case df is returned unchanged.
    """
    if ref_df is None or ref_join_key is None:
        return df, None

    survey_join_key = None
    for candidate in ["customer_account_number", "cas_account_number"]:
        if candidate in df.columns:
            survey_join_key = candidate
            break

    if not survey_join_key:
        print("[REFERENCE] skipping enrichment - neither customer_account_number "
            "nor cas_account_number present in this dataset's columns.")

        return df, None

    ref_value_cols = [c for c in ref_df.columns if c != ref_join_key]
    already_present = [c for c in ref_value_cols if c in df.columns]
    if already_present:
        print(f"[REFERENCE] skippingenrichment - columns {already_present}"
            f"already exist on this dataset, meaning this reference data"
            f"was already joined earlier. Returning dataset unchanged.")
        return df, survey_join_key

    df = df.withColumn(survey_join_key, col(survey_join_key).cast("string"))

    join_source = ref_df.withColumnRenamed(ref_join_key, survey_join_key)

    enriched = df.join(join_source, on=survey_join_key, how="left")
    return enriched, survey_join_key

def get_latest_partition_value(bucket: str, prefix: str, partition_key: str):
    """
    Lists partition-style 'folders' directly under s3://bucket/prefix that
    follow the partition_key= naming convention (e.g. insert_date=2026-07-08/),
    and returns the latest value found (lexical max - correct for YYYY-MM-DD).
    Returns None if no matching partitions exist.
    """
    paginator = s3.get_paginator("list_objects_v2")
    latest_value = None
    marker = f"{partition_key}="
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            folder = cp["Prefix"]
            if marker in folder:
                value = folder.split(marker)[-1].rstrip("/")
                if latest_value is None or value > latest_value:
                    latest_value = value
    return latest_value

# EVENT DATE
try:
    if processing_mode == "manual":
        dates_to_process = ["2026-01-02"]
        print(f"Manual mode, date={dates_to_process}")
    else:
        print("Auto mode: reading event_date from JSON")
        path_clean = EVENT_DATE_PATH.replace("s3://","")
        bucket_name = path_clean.split("/")[0]
        key_name = "/".join(path_clean.split("/")[1:]).rstrip("/") + "/event_dates.json"
        response = s3.get_object(Bucket=bucket_name, Key=key_name)
        event_date_json = json.loads(response["Body"].read().decode("utf-8"))
        ingestion_key = sorted(event_date_json.keys())[-1]
        raw_event_date = event_date_json[ingestion_key]
        dates_to_process = raw_event_date if isinstance(raw_event_date, list) else [raw_event_date]
        print(f"Auto mode, dates to process={dates_to_process}")
except Exception as e:
    logger.error(f"Failed to determine event_date. Error: {str(e)}")
    raise

#Arrears Reference Data - loaded once per job run (slowly-changing billing
# dimension data, not re-read per event_date)
ARREARS_JOIN_KEY = "account_number" #arrears-side column name

arrears_df = None
arrears_join_key = None

try:
    _arrears_raw = spark.read.parquet(ARREARS_S3_PATH)

    if ARREARS_JOIN_KEY not in _arrears_raw.columns:
        print(f"[WARN] Expected join key '{ARREARS_JOIN_KEY} not found in "
            f"{ARREARS_S3_PATH}. Available columns: "
            f"{_arrears_raw.columns}. Arreras enrichment will be skipped.")

    else:
        arrears_join_key = ARREARS_JOIN_KEY
        arrears_df = (
            _arrears_raw
            .withColumn(arrears_join_key, col(arrears_join_key).cast("string"))
            .select(
                col(arrears_join_key).alias(arrears_join_key),
                col("30_day_arrears").alias("arrears_30_day"),
                col("60_day_arrears").alias("arrears_60_day"),
                col("90_day_arrears").alias("arrears_90_day"),
                col("tariff_rate_code").alias("arrears_tariff_rate_code"),
                col("current_balance").alias("arrears_current_balance"),
            )
        )

        print(f"[ARREARS] Loaded {arrears_df.count()} rows from "
            f"{ARREARS_S3_PATH}, join key: {arrears_join_key}")

except Exception as e:
    print(f"[ERROR] Failed to load arrears reference data from {ARREARS_S3_PATH}: {e}")
    arrears_df = None
    arrears_join_key = None

cas_df = None
cas_join_key = None

try:
    latest_insert_date = get_latest_partition_value(CAS_BUCKET, CAS_PREFIX, "insert_date")

    if not latest_insert_date:
        print(f"[WARN] No insert_date partitions found under s3://{CAS_BUCKET}/{CAS_PREFIX}. "
            f"CAS enrichment will be skipped.")
    else:
        CAS_S3_PATH = f"s3://{CAS_BUCKET}/{CAS_PREFIX}insert_date={latest_insert_date}/"
        print(f"[CAS] Reading latest partition: {CAS_S3_PATH}")

        _cas_raw = spark.read.parquet(CAS_S3_PATH)

        if CAS_JOIN_KEY not in _cas_raw.columns:
            print(f"[WARN] Expected join key '{CAS_JOIN_KEY}' not found in "
                f"{CAS_S3_PATH}. Available columns: {_cas_raw.columns}. "
                f"CAS enrichment will be skipped.")
        else:
            cas_join_key = CAS_JOIN_KEY
            cas_df = (
                _cas_raw
                .withColumn(cas_join_key, col(cas_join_key).cast("string"))
                .select(
                    col(cas_join_key).alias(cas_join_key),
                    col("bal_billing_code").alias("cas_bal_billing_code"),
                    col("current_credit_history_code").alias("cas_current_credit_history_code"),
                )
            )
            print(f"[CAS] Loaded {cas_df.count()} rows from {CAS_S3_PATH}, join key: {cas_join_key}")

except Exception as e:
    print(f"[ERROR] Failed to load CAS reference data: {e}")
    cas_df = None
    cas_join_key = None

#For Loop Starts Here
for event_date_value in dates_to_process:
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"[LOOP] Processing event_date: {event_date_value}")
    print(separator)

    # reset all variables
    ivr_final = None
    txn_final = None
    rel_final = None

    # READ ALREADY-CURATED FINAL DATA (already normalized, intent-mapped,
    # and deduped upstream - this job now only enriches and re-writes)
    print("Reading already-curated FINAL datasets")

    print(f"IVR read path: {FINAL_IVR_PATH}event_date={event_date_value}/")
    ivr_final = safe_read_parquet(
        f"{FINAL_IVR_PATH}event_date={event_date_value}/",
        "IVR"
    )
    if ivr_final is not None:
        print(f"[INPUT COUNT] IVR records received: {ivr_final.count()}")
    else:
        print(f"[INPUT COUNT] IVR: No data found / skipped")

    print(f"txn read path: {FINAL_TXN_PATH}event_date={event_date_value}/")
    txn_final = safe_read_parquet(
        f"{FINAL_TXN_PATH}event_date={event_date_value}/",
        "TXN"
    )
    if txn_final is not None:
        print(f"[INPUT COUNT] TXN records received: {txn_final.count()}")
    else:
        print(f"[INPUT COUNT] TXN: No data found / skipped")

    print(f"rel read path: {FINAL_REL_PATH}event_date={event_date_value}/")
    rel_final = safe_read_parquet(
        f"{FINAL_REL_PATH}event_date={event_date_value}/",
        "REL"
    )
    if rel_final is not None:
        print(f"[INPUT COUNT] REL records received: {rel_final.count()}")
    else:
        print(f"[INPUT COUNT] REL: No data found / skipped")

    if ivr_final is None and txn_final is None and rel_final is None:
        print(f"[SKIP] No data for event_date={event_date_value}. Moving to next date.")
        continue

    if ivr_final is not None:
        ivr_final, ivr_arrears_key = enrich_with_reference(ivr_final, arrears_df, arrears_join_key)
        if ivr_arrears_key:
            print(f"[ARREARS] Enriched IVR using join key '{ivr_arrears_key}'")

    if txn_final is not None:
        txn_final, txn_arrears_key = enrich_with_reference(txn_final, arrears_df, arrears_join_key)
        if txn_arrears_key:
            print(f"[ARREARS] Enriched TXN using join key '{txn_arrears_key}'")

    if rel_final is not None:
        rel_final, rel_arrears_key = enrich_with_reference(rel_final, arrears_df, arrears_join_key)
        if rel_arrears_key:
            print(f"[ARREARS] Enriched REL using join key '{rel_arrears_key}'")

    #Add CAS call site
    if ivr_final is not None:
        ivr_final, ivr_cas_key = enrich_with_reference(ivr_final, cas_df, cas_join_key)
        if ivr_cas_key:
            print(f"[CAS] Enriched IVR using join key '{ivr_cas_key}'")

    if txn_final is not None:
        txn_final, txn_cas_key = enrich_with_reference(txn_final, cas_df, cas_join_key)
        if txn_cas_key:
            print(f"[CAS] Enriched TXN using join key '{txn_cas_key}'")

    if rel_final is not None:
        rel_final, rel_cas_key = enrich_with_reference(rel_final, cas_df, cas_join_key)
        if rel_cas_key:
            print(f"[CAS] Enriched REL using join key '{rel_cas_key}'")

    # WRITE CAS-JOIN OUTPUT (partitioned by event_date)

    try:
        print("Writing CAS-join output tables (partitioned by event_date)")

        if ivr_final is not None:
            #write to staging
            ivr_final.write.mode("overwrite").parquet(CAS_JOIN_IVR_STAGING)
            #Read back and write to final partition
            spark.read.parquet(CAS_JOIN_IVR_STAGING).write.mode("overwrite").parquet(
                f"{CAS_JOIN_IVR_PATH}event_date={event_date_value}/"
            )
            # Clean up staging
            try:
                bucket = CAS_JOIN_IVR_STAGING.replace("s3://","").split("/")[0]
                prefix = "/".join(CAS_JOIN_IVR_STAGING.replace("s3://","").split("/")[1:])
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if objs:
                    keys = [{"Key": o["Key"]} for o in objs]
                    s3.delete_objects(Bucket=bucket,Delete={"Objects": keys})
            except Exception:
                pass
            print("IVR CAS-join output written successfully")

        if txn_final is not None:
            #write to staging
            txn_final.write.mode("overwrite").parquet(CAS_JOIN_TXN_STAGING)
            #Read back and write to final partition
            spark.read.parquet(CAS_JOIN_TXN_STAGING).write.mode("overwrite").parquet(
                f"{CAS_JOIN_TXN_PATH}event_date={event_date_value}/"
            )
            # Clean up staging
            try:
                bucket = CAS_JOIN_TXN_STAGING.replace("s3://","").split("/")[0]
                prefix = "/".join(CAS_JOIN_TXN_STAGING.replace("s3://","").split("/")[1:])
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if objs:
                    keys = [{"Key": o["Key"]} for o in objs]
                    s3.delete_objects(Bucket=bucket,Delete={"Objects": keys})
            except Exception:
                pass
            print("TXN CAS-join output written successfully")

        if rel_final is not None:
            #write to staging
            rel_final.write.mode("overwrite").parquet(CAS_JOIN_REL_STAGING)
            #Read back and write to final partition
            spark.read.parquet(CAS_JOIN_REL_STAGING).write.mode("overwrite").parquet(
                f"{CAS_JOIN_REL_PATH}event_date={event_date_value}/"
            )
            # Clean up staging
            try:
                bucket = CAS_JOIN_REL_STAGING.replace("s3://","").split("/")[0]
                prefix = "/".join(CAS_JOIN_REL_STAGING.replace("s3://","").split("/")[1:])
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if objs:
                    keys = [{"Key": o["Key"]} for o in objs]
                    s3.delete_objects(Bucket=bucket,Delete={"Objects": keys})
            except Exception:
                pass
            print("REL CAS-join output written successfully")
    except Exception as e:
        logger.error(f"Failed to write CAS-join parquet files. Error:{str(e)}")
        raise

    ivr_out = ivr_final.count() if ivr_final is not None else "N/A"
    txn_out = txn_final.count() if txn_final is not None else "N/A"
    rel_out = rel_final.count() if rel_final is not None else "N/A"

    print("=" * 60)
    print(f"[FINAL SUMMARY] event_date: {event_date_value}")
    print(f"[FINAL SUMMARY] IVR - CAS-join output rows: {ivr_out}")
    print(f"[FINAL SUMMARY] TXN - CAS-join output rows: {txn_out}")
    print(f"[FINAL SUMMARY] REL - CAS-join output rows: {rel_out}")
    print("=" * 60)

print("Job completed successfully.")
job.commit()