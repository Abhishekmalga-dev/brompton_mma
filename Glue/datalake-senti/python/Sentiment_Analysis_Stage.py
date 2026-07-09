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
for opt in ["mode", "ENV", "STAGING"]:
    if f"--{opt}" in sys.argv:
        optional_args.append(opt)

args = getResolvedOptions(sys.argv, required_args + optional_args)
processing_mode = args.get("mode", "manual").lower()

ENV = args.get("ENV", "dev").lower()
ACCOUNT_TIER = "nonprod" if ENV == "dev" else "prod"

STAGING = args.get("STAGING", "dev")
STAGING_SEGMENT = "" if STAGING == "dev" else STAGING

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
print(f"Processing mode received: {processing_mode}")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3 = boto3.client("s3")

# PATHS

def build_path(bucket: str, *parts) -> str:
    """Builds an s3:// path from a bucket and any number of path segments,
    silently skipping empty segments so we never get a double slash.
    Always returns a trailing slash."""
    clean_parts = [p.strip("/") for p in parts if p]
    return f"s3://{bucket}/" + "/".join(clean_parts) + "/"

def build_prefix(*parts) -> str:
    """Same segment-skipping logic as build_path, but for a bare S3 prefix
    (no bucket, no s3:// scheme) — used where bucket and prefix are passed
    separately to a boto3 call."""
    clean_parts = [p.strip("/") for p in parts if p]
    return "/".join(clean_parts) + "/"

CURATED_BUCKET = f"psegli-datalake{ACCOUNT_TIER}li-datalake-curated-{ENV}"
TEMP_BUCKET = f"psegli-datalake{ACCOUNT_TIER}li-datalake-temp-{ENV}"

EVENT_DATE_PATH = build_path(TEMP_BUCKET, STAGING_SEGMENT, "ccaas/event_dates/survey_api_json")

CURATED_IVR_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "ccaas/survey_customer_sat_ivr")
CURATED_TXN_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "ccaas/survey_sms_web_transactional")
CURATED_REL_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "ccaas/survey_sms_web_relational")

INTENT_MAPPING_PATH = build_path(TEMP_BUCKET, STAGING_SEGMENT, "sentiment_analysis/Intent_Mapping")

FINAL_IVR_STAGING = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/staging/ivr")
FINAL_TXN_STAGING = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/staging/txn")
FINAL_REL_STAGING = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/staging/rel")

FINAL_IVR_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/ivr")
FINAL_TXN_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/transactional")
FINAL_REL_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/final/relational")

# AGG_*_FILE are single-file S3 keys (end in .parquet), not folders.
# build_path() always appends a trailing slash, so we rstrip it back off
# here to preserve the exact original no-trailing-slash file path while
# still deriving ENV/STAGING through the shared helper.
AGG_IVR_FILE = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/aggregate/ivr.parquet").rstrip("/")
AGG_TXN_FILE = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/aggregate/txn.parquet").rstrip("/")
AGG_REL_FILE = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/aggregate/rel.parquet").rstrip("/")

YTD_IVR_FILE = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/ytd/ivr").rstrip("/")
YTD_TXN_FILE = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/ytd/txn").rstrip("/")
YTD_REL_FILE = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/ytd/rel").rstrip("/")

# temp folders for single-file writes
AGG_IVR_TMP = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/aggregate/tmp_ivr")
AGG_TXN_TMP = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/aggregate/tmp_txn")
AGG_REL_TMP = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/aggregate/tmp_rel")

# HELPERS

def write_single_file(df, final_path: str, tmp_path: str, label: str):
    print(f"[{label}] Writing single-file output")

    # write to temp folder
    df.coalesce(1).write.mode("overwrite").parquet(tmp_path)

    tmp_bucket = tmp_path.replace("s3://", "").split("/")[0]
    tmp_prefix = "/".join(tmp_path.replace("s3://", "").split("/")[1:])

    resp = s3.list_objects_v2(Bucket=tmp_bucket, Prefix=tmp_prefix)
    contents = resp.get("Contents", [])
    part_key = None
    for obj in contents:
        if obj["Key"].endswith(".parquet"):
            part_key = obj["Key"]
            break

    if not part_key:
        logger.error(f"[{label}] No parquet part file found in temp path {tmp_path}")
        return

    final_bucket = final_path.replace("s3://", "").split("/")[0]
    final_key = "/".join(final_path.replace("s3://", "").split("/")[1:])

    s3.copy_object(
        Bucket=final_bucket,
        CopySource={"Bucket": tmp_bucket, "Key": part_key},
        Key=final_key
    )

    for obj in contents:
        s3.delete_object(Bucket=tmp_bucket, Key=obj["Key"])

    print(f"[{label}] Written to {final_path}")

def batch_level_dedup(df, dedupe_keys: List[str], ts_cols: List[str]):
    dedupe_keys = [k for k in dedupe_keys if k]
    ts_col = next((c for c in ts_cols if c in df.columns), None)
    if not ts_col:
        return df.dropDuplicates(dedupe_keys)
    w = Window.partitionBy(*dedupe_keys).orderBy(col(ts_col).desc_nulls_last())
    return df.withColumn("_rn", row_number().over(w)).filter(col("_rn") == 1).drop("_rn")

def normalize_ivr(df):
    return df.withColumn("q1_csat_with_ivr", F.col("q1_csat_with_ivr_value").cast(DoubleType())) \
        .withColumn("q2_ease_with_ivr", F.col("q2_ease_with_ivr_value").cast(DoubleType())) \
        .withColumn("q3_csat_with_pseg_li", F.col("q3_csat_with_pseg_li_value").cast(DoubleType()))

def normalize_txn(df):
    return df.withColumn("q1_satisfaction_value", F.col("q1_satisfaction_value_sms").cast(DoubleType())) \
        .withColumn("q2_effort_value", F.col("q2_effort_value_sms").cast(DoubleType())) \
        .withColumn("q3_overall_satisfaction", F.col("q3_overall_satisfaction_value_sms").cast(DoubleType()))

def normalize_rel(df):
    return df.withColumn("q2_satisfaction_value", F.col("q2_satisfaction_value_sms").cast(DoubleType())) \
        .withColumn("q3_effort_value", F.col("q3_effort_value_sms").cast(DoubleType())) \
        .withColumn("q4_overall_satisfaction", F.col("q4_overall_satisfaction_value_sms").cast(DoubleType()))

def apply_intent(df, intent_map_df):
    print("Applying intent mapping")
    intent_map_clean = (
        intent_map_df
        .withColumn("old_intent_clean", lower(trim(F.col("old_intent"))))
        .withColumn("new_intent_clean", lower(trim(F.col("new_intent"))))
        .dropDuplicates(["old_intent_clean"])
    )

    df_clean = df.withColumn(
        "intent_clean",
        lower(trim(F.col("intent")))
    )

    return df_clean.join(
        broadcast(intent_map_clean),
        df_clean["intent_clean"] == intent_map_clean["old_intent_clean"],
        "left"
    ).withColumn(
        "intent",
        F.coalesce(F.col("new_intent_clean"), F.col("intent"))
    ).drop(
        "intent_clean",
        "old_intent_clean",
        "new_intent_clean"
    )

def ensure_event_date(df, event_date_value: str):
    if "event_date" in df.columns:
        return df
    return df.withColumn("event_date", F.lit(event_date_value).cast(DateType()))

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

# EVENT DATE
try:
    if processing_mode == "manual":
        dates_to_process = ["2026-07-01","2026-07-02","2026-07-03","2026-07-04","2026-07-05","2026-07-06","2026-07-07"]
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

try:
    print("Reading intent mapping file")
    intent_map_df = spark.read.option("header", "true").csv(INTENT_MAPPING_PATH)

except Exception as e:
    logger.error(f"failed to read intent mapping file from {INTENT_MAPPING_PATH}. Error: {str(e)}")
    raise

#For Loop Starts Here
for event_date_value in dates_to_process:
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"[LOOP] Processing event_date: {event_date_value}")
    print(separator)

    # reset all variables
    ivr_raw = None
    txn_raw = None
    rel_raw = None
    ivr_final = None
    txn_final = None
    rel_final = None

    # READ CURATED
    print("Reading available curated datasets")

    print(f"IVR read path: {CURATED_IVR_PATH}event_date={event_date_value}/")
    ivr_raw = safe_read_parquet(
        f"{CURATED_IVR_PATH}event_date={event_date_value}/",
        "IVR"
    )
    if ivr_raw is not None:
        print(f"[INPUT COUNT] IVR records received: {ivr_raw.count()}")
    else:
        print(f"[INPUT COUNT] IVR: No data found / skipped")

    print(f"txn read path: {CURATED_TXN_PATH}event_date={event_date_value}/")
    txn_raw = safe_read_parquet(
        f"{CURATED_TXN_PATH}event_date={event_date_value}/",
        "TXN"
    )
    if txn_raw is not None:
        print(f"[INPUT COUNT] TXN records received: {txn_raw.count()}")
    else:
        print(f"[INPUT COUNT] TXN: No data found / skipped")

    print(f"rel read path: {CURATED_REL_PATH}event_date={event_date_value}/")
    rel_raw = safe_read_parquet(
        f"{CURATED_REL_PATH}event_date={event_date_value}/",
        "REL"
    )
    if rel_raw is not None:
        print(f"[INPUT COUNT] REL records received: {rel_raw.count()}")
    else:
        print(f"[INPUT COUNT] REL: No data found / skipped")

    if ivr_raw is None and txn_raw is None and rel_raw is None:
        print(f"[SKIP] No data for event_date={event_date_value}. Moving to next date.")
        continue

    # NORMALIZE + INTENT

    if ivr_raw is not None:
        ivr_final = normalize_ivr(ivr_raw)
        ivr_final = apply_intent(ivr_final, intent_map_df)
        ivr_final = ensure_event_date(ivr_final, event_date_value)
        #ivr_final = dedupe_ivr_specific(ivr_final)
        print(f"[PROCESSED COUNT] IVR records after normalize+intent+dedup: {ivr_final.count()}")
    else:
        logger.info("IVR not available. Skipping IVR processing")
    if txn_raw is not None:
        txn_final = normalize_txn(txn_raw)
        txn_final = apply_intent(txn_final, intent_map_df)
        txn_final = ensure_event_date(txn_final, event_date_value)
        print(f"[PROCESSED COUNT] TXN records after normalize+intent+dedup: {txn_final.count()}")
    else:
        logger.info("TXN not available. Skipping TXN processing")

    if rel_raw is not None:
        rel_final = normalize_rel(rel_raw)
        rel_final = apply_intent(rel_final, intent_map_df)
        rel_final = ensure_event_date(rel_final, event_date_value)
        print(f"[PROCESSED COUNT] REL records after normalize+intent+dedup: {rel_final.count()}")
    else:
        logger.info("REL not available. Skipping REL processing")

    # DEDUPE

    dedupe_keys = [
        "response_received_date",
        "cas_account_number",
        "survey_id",
        "survey_name",
        "case_type",
        "event_ts",
        "queue_time"
    ]

    ts_cols = ["response_received_date", "event_ts"]

    if ivr_final is not None:
        ivr_final = batch_level_dedup(ivr_final, dedupe_keys, ts_cols)
    if txn_final is not None:
        txn_final = batch_level_dedup(txn_final, dedupe_keys, ts_cols)
    if rel_final is not None:
        rel_final = batch_level_dedup(rel_final, dedupe_keys, ts_cols)

    print(f"IVR final count after dedupe: {ivr_final.count() if ivr_final is not None else 'N/A'}")
    print(f"TXN final count after dedupe: {txn_final.count() if txn_final is not None else 'N/A'}")
    print(f"REL final count after dedupe: {rel_final.count() if rel_final is not None else 'N/A'}")

    # WRITE FINAL (partitioned by event_date)

    try:
        print("Writing FINAL tables (partitioned by event_date)")

        if ivr_final is not None:
            #write to staging
            ivr_final.write.mode("overwrite").parquet(FINAL_IVR_STAGING)
            #Read back and write to final partition
            spark.read.parquet(FINAL_IVR_STAGING).write.mode("overwrite").parquet(
                f"{FINAL_IVR_PATH}event_date={event_date_value}/"
            )
            # Clean up staging
            try:
                bucket = FINAL_IVR_STAGING.replace("s3://","").split("/")[0]
                prefix = "/".join(FINAL_IVR_STAGING.replace("s3://","").split("/")[1:])
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if objs:
                    keys = [{"key": o["key"]} for o in objs]
                    s3.delete_objects(Bucket=bucket,Delete={"Objects": keys})
            except Exception:
                pass
            print("IVR final written successfully")

        if txn_final is not None:
            #write to staging
            txn_final.write.mode("overwrite").parquet(FINAL_TXN_STAGING)
            #Read back and write to final partition
            spark.read.parquet(FINAL_TXN_STAGING).write.mode("overwrite").parquet(
                f"{FINAL_TXN_PATH}event_date={event_date_value}/"
            )
            # Clean up staging
            try:
                bucket = FINAL_TXN_STAGING.replace("s3://","").split("/")[0]
                prefix = "/".join(FINAL_TXN_STAGING.replace("s3://","").split("/")[1:])
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if objs:
                    keys = [{"key": o["key"]} for o in objs]
                    s3.delete_objects(Bucket=bucket,Delete={"Objects": keys})
            except Exception:
                pass
            print("TXN final written successfully")

        if rel_final is not None:
            #write to staging
            rel_final.write.mode("overwrite").parquet(FINAL_REL_STAGING)
            #Read back and write to final partition
            spark.read.parquet(FINAL_REL_STAGING).write.mode("overwrite").parquet(
                f"{FINAL_REL_PATH}event_date={event_date_value}/"
            )
            # Clean up staging
            try:
                bucket = FINAL_REL_STAGING.replace("s3://","").split("/")[0]
                prefix = "/".join(FINAL_REL_STAGING.replace("s3://","").split("/")[1:])
                objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if objs:
                    keys = [{"key": o["key"]} for o in objs]
                    s3.delete_objects(Bucket=bucket,Delete={"Objects": keys})
            except Exception:
                pass
            print("REL final written successfully")
    except Exception as e:
        logger.error(f"Failed to write final parquet files. Error:{str(e)}")
        raise

    # DAILY AGGREGATES

    print("Computing daily aggregates")
    daily_date = F.lit(event_date_value).cast(DateType())

    ivr_agg = None
    txn_agg = None
    rel_agg = None

    if ivr_final is not None:
        ivr_agg = ivr_final.filter(F.col("event_date") == daily_date).agg(
            F.avg("q1_csat_with_ivr").alias("q1_avg_csat"),
            F.avg("q2_ease_with_ivr").alias("q2_avg_csat"),
            F.avg("q3_csat_with_pseg_li").alias("q3_avg_csat")
        ).withColumn("event_date", daily_date)

    if txn_final is not None:
        txn_agg = txn_final.filter(F.col("event_date") == daily_date).agg(
            F.avg("q1_satisfaction_value").alias("q1_avg_csat"),
            F.avg("q2_effort_value").alias("q2_avg_csat"),
            F.avg("q3_overall_satisfaction").alias("q3_avg_csat")
        ).withColumn("event_date", daily_date)

    if rel_final is not None:
        rel_agg = rel_final.filter(F.col("event_date") == daily_date).agg(
            F.avg("q2_satisfaction_value").alias("q1_avg_csat"),
            F.avg("q3_effort_value").alias("q2_avg_csat"),
            F.avg("q4_overall_satisfaction").alias("q3_avg_csat")
        ).withColumn("event_date", daily_date)

    # YTD AGGREGATES

    print("Computing YTD aggregates")
    today = F.current_date()

    ivr_ytd = None
    txn_ytd = None
    rel_ytd = None

    if ivr_final is not None:
        ivr_ytd = ivr_final.filter(
            F.col("event_date") == daily_date
        ).agg(
            F.avg("q1_csat_with_ivr").alias("q1_avg_csat"),
            F.avg("q2_ease_with_ivr").alias("q2_avg_csat"),
            F.avg("q3_csat_with_pseg_li").alias("q3_avg_csat")
        ).withColumn("event_date", daily_date) \
         .withColumn("processing_date", today)

    if txn_final is not None:
        txn_ytd = txn_final.filter(
            F.col("event_date") == daily_date
        ).agg(
            F.avg("q1_satisfaction_value").alias("q1_avg_csat"),
            F.avg("q2_effort_value").alias("q2_avg_csat"),
            F.avg("q3_overall_satisfaction").alias("q3_avg_csat")
        ).withColumn("event_date", daily_date) \
         .withColumn("processing_date", today)

    if rel_final is not None:
        rel_ytd = rel_final.filter(
            F.col("event_date") == daily_date
        ).agg(
            F.avg("q2_satisfaction_value").alias("q1_avg_csat"),
            F.avg("q3_effort_value").alias("q2_avg_csat"),
            F.avg("q4_overall_satisfaction").alias("q3_avg_csat")
        ).withColumn("event_date", daily_date) \
         .withColumn("processing_date", today)

    # WRITE AGG (one file per survey type)

    try:
        print("Writing aggregate files")
        if ivr_agg is not None:
            write_single_file(ivr_agg, AGG_IVR_FILE, AGG_IVR_TMP, "IVR AGG")
        if txn_agg is not None:
            write_single_file(txn_agg, AGG_TXN_FILE, AGG_TXN_TMP, "TXN AGG")
        if rel_agg is not None:
            write_single_file(rel_agg, AGG_REL_FILE, AGG_REL_TMP, "REL AGG")
        print("Aggregate files written successfully")
    except Exception as e:
        logger.error(f"Failed to write AGG files. Error: {str(e)}")
        raise

    # WRITE YTD (one file per survey type)

    try:
        print("Writing YTD files")
        if ivr_ytd is not None:
            (
                ivr_ytd.write
                .mode("append")
                .partitionBy("event_date")
                .parquet(YTD_IVR_FILE)
            )
            print("IVR YTD written successfully")
        if txn_ytd is not None:
            (
                txn_ytd.write
                .mode("append")
                .partitionBy("event_date")
                .parquet(YTD_TXN_FILE)
            )
            print("TXN YTD written successfully")
        if rel_ytd is not None:
            (
                rel_ytd.write
                .mode("append")
                .partitionBy("event_date")
                .parquet(YTD_REL_FILE)
            )
            print("REL YTD written successfully")
    except Exception as e:
        logger.error(f"Failed to write YTD files. Error: {str(e)}")
        raise

    ivr_in = ivr_raw.count() if ivr_raw is not None else "N/A"
    ivr_out = ivr_final.count() if ivr_final is not None else "N/A"
    txn_in = txn_raw.count() if txn_raw is not None else "N/A"
    txn_out = txn_final.count() if txn_final is not None else "N/A"
    rel_in = rel_raw.count() if rel_raw is not None else "N/A"
    rel_out = rel_final.count() if rel_final is not None else "N/A"

    print("=" * 60)
    print(f"[FINAL SUMMARY] event_date: {event_date_value}")
    print(f"[FINAL SUMMARY] IVR - Input: {ivr_in} | Final: {ivr_out}")
    print(f"[FINAL SUMMARY] TXN - Input: {txn_in} | Final: {txn_out}")
    print(f"[FINAL SUMMARY] REL - Input: {rel_in} | Final: {rel_out}")
    print("=" * 60)

print("Job completed successfully")
job.commit()