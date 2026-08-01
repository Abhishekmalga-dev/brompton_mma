import sys
import boto3, json
from datetime import date
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

# Force modern TIMESTAMP_MICROS parquet encoding for this job's own writes
# (FINAL_*_PATH, AGG_*_FILE, YTD_*_FILE). Without this, Spark defaults to
# the legacy INT96 timestamp encoding, which readers that don't specifically
# decode INT96 (S3 Select, some downstream jobs relying on a Glue Catalog
# schema) will misread as garbage scientific-notation numbers -- e.g.
# response_received_date, event_ts, queue_time, all passed through from the
# curated read into ivr_final/txn_final/rel_final and written back out here.
# NOTE: this only affects NEW writes going forward. It does not retroactively
# fix already-written historical partitions under FINAL_*_PATH -- those
# still need a separate read+rewrite pass, same as the upstream job.
spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")

logger = glueContext.get_logger()
print(f"Processing mode received: {processing_mode}")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# Tracker table this job now writes to directly (execution_date partition key,
# one record per day). Previously written by the downstream curation job;
# that write is being removed there so this job is the sole writer going
# forward. Best-supported inference from prior naming conventions
# ("datalake-{service}-{purpose}-{env}") — please confirm/correct if this
# isn't the exact existing table name.
TRACKER_TABLE_NAME = f"datalake-sentiment-analysis-tracker-{ENV}"

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

# Removed-records CSV output (campaign filter + dedupe), one file per
# survey type per event_date.
REMOVED_RECORDS_IVR_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/removed_records/ivr")
REMOVED_RECORDS_TXN_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/removed_records/txn")
REMOVED_RECORDS_REL_PATH = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/removed_records/rel")

REMOVED_RECORDS_IVR_TMP = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/removed_records/tmp_ivr")
REMOVED_RECORDS_TXN_TMP = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/removed_records/tmp_txn")
REMOVED_RECORDS_REL_TMP = build_path(CURATED_BUCKET, STAGING_SEGMENT, "sentiment_analysis/removed_records/tmp_rel")

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

def write_removed_records_csv(df, final_path: str, tmp_path: str, label: str):
    """Same staging/copy/cleanup pattern as write_single_file, adapted for a
    single-file CSV with a header row instead of parquet."""
    print(f"[{label}] Writing removed-records CSV output")

    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_path)

    tmp_bucket = tmp_path.replace("s3://", "").split("/")[0]
    tmp_prefix = "/".join(tmp_path.replace("s3://", "").split("/")[1:])

    resp = s3.list_objects_v2(Bucket=tmp_bucket, Prefix=tmp_prefix)
    contents = resp.get("Contents", [])
    part_key = None
    for obj in contents:
        if obj["Key"].endswith(".csv"):
            part_key = obj["Key"]
            break

    if not part_key:
        logger.error(f"[{label}] No CSV part file found in temp path {tmp_path}")
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

    print(f"[{label}] Removed-records CSV written to {final_path}")

def batch_level_dedup(df, dedupe_keys: List[str], ts_cols: List[str]):
    """
    Returns (deduped, removed), both derived from the SAME ranked DataFrame
    via _rn filtering -- NOT via exceptAll. exceptAll between two separately
    derived DataFrames that both trace through apply_intent()'s broadcast
    join can trigger Spark attribute-resolution errors (Catalyst loses track
    of which "new_intent" column, by internal expression ID, is being
    referenced -- "Couldn't find new_intent#N in [...]"). Deriving both
    outputs from one ranked DataFrame avoids that entirely, matching the
    approach already used safely in dedupe_txn_rel_specific.
    """
    dedupe_keys = [k for k in dedupe_keys if k]
    ts_col = next((c for c in ts_cols if c in df.columns), None)
    order_col = col(ts_col).desc_nulls_last() if ts_col else F.lit(1)
    w = Window.partitionBy(*dedupe_keys).orderBy(order_col)
    ranked = df.withColumn("_rn", row_number().over(w))
    deduped = ranked.filter(col("_rn") == 1).drop("_rn")
    removed = ranked.filter(col("_rn") > 1).drop("_rn")
    return deduped, removed

def dedupe_txn_rel_specific(df, label: str, comment_cols: List[str]):
    """
    TXN/REL specific dedup logic based on response_received_date, cas_account_number,
    and comment (coalesced from the survey-specific comment_cols, e.g.
    q4_comment_in_survey_language_sms / q4_comment_sms for TXN, or
    q5_comment_in_survey_language_sms / q5_comment_sms for REL).

    Case 1: response_received_date + cas_account_number + comment all match -> duplicate.
    Case 2: comment is null, response_received_date + cas_account_number match -> duplicate.
    Case 3: cas_account_number + comment match but response_received_date differs -> duplicate
            (cross-date duplicate for the same account/comment).

    Cases 1 and 3 use the same underlying dedup key (cas_account_number + comment,
    date-agnostic) — they are only split apart after the fact, for reporting, into
    "same date" (Case 1) vs "different date" (Case 3) duplicates. Case 2 uses a
    separate key since there is no comment to key off of.
    The latest response_received_date is kept within each duplicate group.
    """
    total_before = df.count()

    comment_source_cols = [c for c in comment_cols if c in df.columns]
    if comment_source_cols:
        comment_expr = F.coalesce(*[F.col(c) for c in comment_source_cols])
    else:
        comment_expr = F.lit(None).cast("string")

    df = df.withColumn("_comment_value", comment_expr)

    has_comment_df = df.filter(F.col("_comment_value").isNotNull())
    no_comment_df = df.filter(F.col("_comment_value").isNull())

    has_comment_total = has_comment_df.count()
    no_comment_total = no_comment_df.count()

    # CASE 2: comment is null -> dedupe on response_received_date + cas_account_number
    w_case2 = Window.partitionBy("response_received_date", "cas_account_number") \
        .orderBy(F.col("response_received_date").asc_nulls_last())
    no_comment_ranked = no_comment_df.withColumn("_rn", row_number().over(w_case2))
    case2_duplicates = no_comment_ranked.filter(F.col("_rn") > 1)
    case2_dup_count = case2_duplicates.count()
    no_comment_deduped = no_comment_ranked.filter(F.col("_rn") == 1).drop("_rn")

    # CASE 1 + CASE 3: comment present -> dedupe on cas_account_number + comment (date-agnostic)
    w_group = Window.partitionBy("cas_account_number", "_comment_value")
    w_rank = w_group.orderBy(F.col("response_received_date").desc_nulls_last())
    has_comment_ranked = has_comment_df \
        .withColumn("_rn", row_number().over(w_rank)) \
        .withColumn("_kept_date", F.first("response_received_date").over(w_rank))

    case13_duplicates = has_comment_ranked.filter(F.col("_rn") > 1)
    case1_duplicates = case13_duplicates.filter(F.col("response_received_date") == F.col("_kept_date"))
    case3_duplicates = case13_duplicates.filter(F.col("response_received_date") != F.col("_kept_date"))
    case1_dup_count = case1_duplicates.count()
    case3_dup_count = case3_duplicates.count()

    has_comment_deduped = has_comment_ranked.filter(F.col("_rn") == 1).drop("_rn", "_kept_date")

    final_df = no_comment_deduped.unionByName(has_comment_deduped).drop("_comment_value")
    total_after = final_df.count()

    # VALIDATION REPORT
    sample_cols = [c for c in ["response_received_date", "cas_account_number"] + comment_cols if c in df.columns]

    print("#" * 60)
    print(f"{label} DEDUP VALIDATION REPORT")
    print("#" * 60)
    print(f"{label} - Total Input Rows              : {total_before}")
    print(f"{label} - Records WITH comment           : {has_comment_total}")
    print(f"{label} - Records WITHOUT comment        : {no_comment_total}")
    print("-" * 60)
    print(f"{label} - Case 1 Duplicates Found        : {case1_dup_count}")
    print(f"{label} - Case 2 Duplicates Found        : {case2_dup_count}")
    print(f"{label} - Case 3 Duplicates Found        : {case3_dup_count}")
    print(f"{label} - Record Count After Dedup       : {total_after}")
    print("-" * 60)

    if case1_dup_count > 0:
        print(f"{label} - Case 1 Sample Duplicates (response_received_date, cas_account_number, comment same):")
        case1_duplicates.select(*sample_cols).show(5, truncate=False)
    if case2_dup_count > 0:
        print(f"{label} - Case 2 Sample Duplicates (comment null, response_received_date + cas_account_number same):")
        case2_duplicates.select(*sample_cols).show(5, truncate=False)
    if case3_dup_count > 0:
        print(f"{label} - Case 3 Sample Duplicates (cas_account_number + comment same, response_received_date differs):")
        case3_duplicates.select(*sample_cols).show(5, truncate=False)

    print("#" * 60)

    # Build a combined removed-records dataframe (all 3 cases, tagged with
    # which case caused removal), for CSV export purposes.
    removed_cols = ["response_received_date", "cas_account_number"]
    case1_tagged = case1_duplicates.select(*removed_cols).withColumn("removal_reason", F.lit("DEDUP_CASE_1"))
    case2_tagged = case2_duplicates.select(*removed_cols).withColumn("removal_reason", F.lit("DEDUP_CASE_2"))
    case3_tagged = case3_duplicates.select(*removed_cols).withColumn("removal_reason", F.lit("DEDUP_CASE_3"))
    all_removed = case1_tagged.unionByName(case2_tagged).unionByName(case3_tagged)

    return final_df, all_removed

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
        "new_intent_clean",
        "old_intent"
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

# Accumulators for the tracker-table update at the end of the job: every
# event_date processed in this run, plus per-date/per-source new-records info.
all_dates_processed = []
source_new_records_report = []

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
    ivr_campaign_removed = 0
    ivr_new_records_found = False
    txn_new_records_found = False
    rel_new_records_found = False
    ivr_campaign_removed_df = None
    ivr_dedup_removed = None
    txn_dedup_removed = None
    rel_dedup_removed = None

    # READ CURATED
    print("Reading available curated datasets")

    print(f"IVR read path: {CURATED_IVR_PATH}event_date={event_date_value}/")
    ivr_raw = safe_read_parquet(
        f"{CURATED_IVR_PATH}event_date={event_date_value}/",
        "IVR"
    )
    if ivr_raw is not None:
        print(f"[INPUT COUNT] IVR records received: {ivr_raw.count()}")
        ivr_verbatim_cols = [c for c in ["q5_overall_exp_comment_in_survey_language", "q5_overall_exp_comment"] if c in ivr_raw.columns]
        if ivr_verbatim_cols:
            ivr_verbatim_count = ivr_raw.filter(F.coalesce(*[F.col(c) for c in ivr_verbatim_cols]).isNotNull()).count()
        else:
            ivr_verbatim_count = 0
        print(f"[INPUT COUNT] IVR records WITH verbatim: {ivr_verbatim_count}")

        # CAMPAIGN FILTER: keep only "Main" campaign records
        ivr_pre_campaign_count = ivr_raw.count()
        if "campaign" in ivr_raw.columns:
            ivr_campaign_removed_df = ivr_raw.filter(F.col("campaign") != "Main")
            ivr_raw = ivr_raw.filter(F.col("campaign") == "Main")
        ivr_post_campaign_count = ivr_raw.count()
        ivr_campaign_removed = ivr_pre_campaign_count - ivr_post_campaign_count
        print(f"[CAMPAIGN FILTER] IVR records removed (non-Main campaign): {ivr_campaign_removed}")
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
        all_dates_processed.append(event_date_value)
        source_new_records_report.append({
            "event_date": event_date_value,
            "ivr_new_records_found": False,
            "sms_web_transactional_new_records_found": False,
            "sms_web_relational_new_records_found": False,
        })
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
        ivr_final, ivr_dedup_removed = batch_level_dedup(ivr_final, dedupe_keys, ts_cols)
    if txn_final is not None:
        txn_final, txn_dedup_removed = dedupe_txn_rel_specific(
            txn_final, "TXN",
            comment_cols=["q4_comment_in_survey_language_sms", "q4_comment_sms"]
        )
    if rel_final is not None:
        rel_final, rel_dedup_removed = dedupe_txn_rel_specific(
            rel_final, "REL",
            comment_cols=["q5_comment_in_survey_language_sms", "q5_comment_sms"]
        )

    print(f"IVR final count after dedupe: {ivr_final.count() if ivr_final is not None else 'N/A'}")
    print(f"TXN final count after dedupe: {txn_final.count() if txn_final is not None else 'N/A'}")
    print(f"REL final count after dedupe: {rel_final.count() if rel_final is not None else 'N/A'}")

    # REMOVED RECORDS CSV: one file per survey type per event_date, combining
    # campaign-filter removals (IVR only) and dedupe removals (all 3), with a
    # removal_reason column to distinguish. Uses cas_account_number as the
    # key identifier, per requirement.
    try:
        removed_cols = ["response_received_date", "cas_account_number"]

        ivr_removed_parts = []
        if ivr_campaign_removed_df is not None and ivr_campaign_removed_df.count() > 0:
            ivr_removed_parts.append(
                ivr_campaign_removed_df.select(*[c for c in removed_cols if c in ivr_campaign_removed_df.columns])
                .withColumn("removal_reason", F.lit("CAMPAIGN_FILTER_NON_MAIN"))
            )
        if ivr_dedup_removed is not None and ivr_dedup_removed.count() > 0:
            ivr_removed_parts.append(
                ivr_dedup_removed.select(*[c for c in removed_cols if c in ivr_dedup_removed.columns])
                .withColumn("removal_reason", F.lit("DEDUP"))
            )
        if ivr_removed_parts:
            ivr_removed_combined = ivr_removed_parts[0]
            for part in ivr_removed_parts[1:]:
                ivr_removed_combined = ivr_removed_combined.unionByName(part)
            ivr_removed_combined = ivr_removed_combined \
                .withColumn("survey_type", F.lit("IVR")) \
                .withColumn("event_date", F.lit(event_date_value))
            write_removed_records_csv(
                ivr_removed_combined,
                f"{REMOVED_RECORDS_IVR_PATH}event_date={event_date_value}/removed_records.csv",
                REMOVED_RECORDS_IVR_TMP,
                "IVR REMOVED RECORDS"
            )

        if txn_dedup_removed is not None and txn_dedup_removed.count() > 0:
            txn_removed_combined = txn_dedup_removed \
                .withColumn("survey_type", F.lit("TXN")) \
                .withColumn("event_date", F.lit(event_date_value))
            write_removed_records_csv(
                txn_removed_combined,
                f"{REMOVED_RECORDS_TXN_PATH}event_date={event_date_value}/removed_records.csv",
                REMOVED_RECORDS_TXN_TMP,
                "TXN REMOVED RECORDS"
            )

        if rel_dedup_removed is not None and rel_dedup_removed.count() > 0:
            rel_removed_combined = rel_dedup_removed \
                .withColumn("survey_type", F.lit("REL")) \
                .withColumn("event_date", F.lit(event_date_value))
            write_removed_records_csv(
                rel_removed_combined,
                f"{REMOVED_RECORDS_REL_PATH}event_date={event_date_value}/removed_records.csv",
                REMOVED_RECORDS_REL_TMP,
                "REL REMOVED RECORDS"
            )
    except Exception as e:
        logger.error(f"Failed to write removed-records CSV files for event_date={event_date_value}. Error: {str(e)}")
        raise

    # WRITE FINAL (partitioned by event_date)

    try:
        print("Writing FINAL tables (partitioned by event_date)")

        if ivr_final is not None:
            # NEW RECORDS CHECK: compare what's already in S3 for this partition
            # against what we're about to write, before the overwrite happens.
            ivr_final_partition = f"{FINAL_IVR_PATH}event_date={event_date_value}/"
            try:
                ivr_existing_count = spark.read.parquet(ivr_final_partition.rstrip("/")).count()
            except Exception:
                ivr_existing_count = 0
            ivr_new_count = ivr_final.count()
            ivr_new_records_found = ivr_new_count > ivr_existing_count
            print(f"[NEW RECORDS CHECK] IVR - Existing in S3 before this run: {ivr_existing_count} | To be written: {ivr_new_count} | New records found: {ivr_new_records_found}")

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
            print(f"[FINAL WRITTEN COUNT] IVR final written output: {ivr_final.count()}")

        if txn_final is not None:
            txn_final_partition = f"{FINAL_TXN_PATH}event_date={event_date_value}/"
            try:
                txn_existing_count = spark.read.parquet(txn_final_partition.rstrip("/")).count()
            except Exception:
                txn_existing_count = 0
            txn_new_count = txn_final.count()
            txn_new_records_found = txn_new_count > txn_existing_count
            print(f"[NEW RECORDS CHECK] TXN - Existing in S3 before this run: {txn_existing_count} | To be written: {txn_new_count} | New records found: {txn_new_records_found}")

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
            rel_final_partition = f"{FINAL_REL_PATH}event_date={event_date_value}/"
            try:
                rel_existing_count = spark.read.parquet(rel_final_partition.rstrip("/")).count()
            except Exception:
                rel_existing_count = 0
            rel_new_count = rel_final.count()
            rel_new_records_found = rel_new_count > rel_existing_count
            print(f"[NEW RECORDS CHECK] REL - Existing in S3 before this run: {rel_existing_count} | To be written: {rel_new_count} | New records found: {rel_new_records_found}")

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

    # YTD AGGREGATES (inside loop, one row per date per survey type)

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
    print(f"[FINAL SUMMARY] IVR - Removed by campaign filter (non-Main): {ivr_campaign_removed}")
    print(f"[FINAL SUMMARY] TXN - Input: {txn_in} | Final: {txn_out}")
    print(f"[FINAL SUMMARY] REL - Input: {rel_in} | Final: {rel_out}")
    print("=" * 60)

    all_dates_processed.append(event_date_value)
    source_new_records_report.append({
        "event_date": event_date_value,
        "ivr_new_records_found": ivr_new_records_found,
        "sms_web_transactional_new_records_found": txn_new_records_found,
        "sms_web_relational_new_records_found": rel_new_records_found,
    })

# DAILY AGGREGATES (computed once after all dates processed)
# This is a cumulative/rolling average across the CURRENT CALENDAR YEAR's data
# only (resets every January 1st), then overwrites the daily aggregate files.
# Schema: event_start_date (earliest event_date in the current year's data),
# event_end_date (latest event_date in the current year's data),
# calculated_date (today, i.e. when this aggregate was computed) — plus the
# avg score columns.

print("Computing daily aggregates (cumulative across current calendar year)")
calculated_date = F.lit(date.today()).cast(DateType())
current_year = date.today().year

ivr_agg = None
txn_agg = None
rel_agg = None

try:
    # Read entire FINAL_IVR_PATH, then scope to the current calendar year
    if ivr_final is not None:
        ivr_all_history = spark.read.parquet(FINAL_IVR_PATH.rstrip("/")) \
            .filter(F.year(F.col("event_date")) == current_year)
        ivr_agg = ivr_all_history.agg(
            F.avg("q1_csat_with_ivr").alias("q1_avg_csat"),
            F.avg("q2_ease_with_ivr").alias("q2_avg_csat"),
            F.avg("q3_csat_with_pseg_li").alias("q3_avg_csat"),
            F.min("event_date").alias("event_start_date"),
            F.max("event_date").alias("event_end_date")
        ).withColumn("calculated_date", calculated_date)
        print(f"[DAILY AGG] IVR: read {ivr_all_history.count()} records for year {current_year}, computed cumulative average")

    # Read entire FINAL_TXN_PATH, then scope to the current calendar year
    if txn_final is not None:
        txn_all_history = spark.read.parquet(FINAL_TXN_PATH.rstrip("/")) \
            .filter(F.year(F.col("event_date")) == current_year)
        txn_agg = txn_all_history.agg(
            F.avg("q1_satisfaction_value").alias("q1_avg_csat"),
            F.avg("q2_effort_value").alias("q2_avg_csat"),
            F.avg("q3_overall_satisfaction").alias("q3_avg_csat"),
            F.min("event_date").alias("event_start_date"),
            F.max("event_date").alias("event_end_date")
        ).withColumn("calculated_date", calculated_date)
        print(f"[DAILY AGG] TXN: read {txn_all_history.count()} records for year {current_year}, computed cumulative average")

    # Read entire FINAL_REL_PATH, then scope to the current calendar year
    if rel_final is not None:
        rel_all_history = spark.read.parquet(FINAL_REL_PATH.rstrip("/")) \
            .filter(F.year(F.col("event_date")) == current_year)
        rel_agg = rel_all_history.agg(
            F.avg("q2_satisfaction_value").alias("q1_avg_csat"),
            F.avg("q3_effort_value").alias("q2_avg_csat"),
            F.avg("q4_overall_satisfaction").alias("q3_avg_csat"),
            F.min("event_date").alias("event_start_date"),
            F.max("event_date").alias("event_end_date")
        ).withColumn("calculated_date", calculated_date)
        print(f"[DAILY AGG] REL: read {rel_all_history.count()} records for year {current_year}, computed cumulative average")

    # WRITE AGG (one file per survey type, overwrite mode)
    print("Writing daily aggregate files (overwrite)")
    if ivr_agg is not None:
        write_single_file(ivr_agg, AGG_IVR_FILE, AGG_IVR_TMP, "IVR AGG")
    if txn_agg is not None:
        write_single_file(txn_agg, AGG_TXN_FILE, AGG_TXN_TMP, "TXN AGG")
    if rel_agg is not None:
        write_single_file(rel_agg, AGG_REL_FILE, AGG_REL_TMP, "REL AGG")
    print("Daily aggregate files written successfully")

except Exception as e:
    logger.error(f"Failed to compute/write daily aggregates. Error: {str(e)}")
    raise

# TRACKER TABLE WRITE: this job is now the sole writer of the per-day tracker
# record (the downstream curation job's DynamoDB write is being removed).
# Availability flags use the same "post-dedup dataframe has >0 rows, False if
# None" definition previously used by the curation job, based on the last
# event_date processed in the loop above.
execution_date_today = date.today().isoformat()

ivr_available = ivr_final is not None and ivr_final.count() > 0
txn_available = txn_final is not None and txn_final.count() > 0
rel_available = rel_final is not None and rel_final.count() > 0

try:
    print(f"[TRACKER] Writing tracker record for execution_date={execution_date_today}")
    tracker_table = dynamodb.Table(TRACKER_TABLE_NAME)
    tracker_table.put_item(
        Item={
            "execution_date": execution_date_today,
            "ivr_available": ivr_available,
            "sms_web_relational_available": rel_available,
            "sms_web_transactional_available": txn_available,
            "dates_processed": all_dates_processed,
            "source_new_records": source_new_records_report,
        }
    )
    print(f"[TRACKER] ivr_available={ivr_available} | sms_web_relational_available={rel_available} | sms_web_transactional_available={txn_available}")
    print(f"[TRACKER] dates_processed={all_dates_processed}")
    print(f"[TRACKER] source_new_records={source_new_records_report}")
except Exception as e:
    logger.error(f"Failed to write tracker record to {TRACKER_TABLE_NAME} for execution_date={execution_date_today}. Error: {str(e)}")
    raise

print("Job completed successfully")
job.commit()