required_args = ["JOB_NAME"]
optional_args = []
for opt in ["mode", "ENV", "ARREARS_TABLE", "ARREARS_JOIN_KEY"]:
    if f"--{opt}" in sys.argv:
        optional_args.append(opt)

args = getResolvedOptions(sys.argv, required_args + optional_args)
processing_mode = args.get("mode", "manual").lower()
ENV = args.get("ENV", "dev").lower()
ARREARS_TABLE = args.get("ARREARS_TABLE", "cas_arrears")
ARREARS_JOIN_KEY = args.get("ARREARS_JOIN_KEY", "account_number")

----
# PATHS

ARREARS_DATABASE = f"datalake-curated-{ENV}"

EVENT_DATE_PATH = "s3://psegli-datalakenonprodli-datalake-temp-dev/ccaas/event_dates/survey_api_json/"

----
#Arrears Reference Data - loaded once per job run (slowly-changing billing
# dimension data, not re-read per event_date). ARREARS_DATABASE,
# ARREARS_TABLE, and ARREARS_JOIN_KEY are declared earlier from job
# parameters/ENV, not hardcoded here.

arrears_df = None
----


# PATHS

ACCOUNT_TIER = "nonprod" if ENV == "dev" else "prod"
TEMP_BUCKET = f"psegli-datalake{ACCOUNT_TIER}li-datalake-temp-{ENV}"
CURATED_BUCKET = f"psegli-datalake{ACCOUNT_TIER}li-datalake-curated-{ENV}"

ARREARS_DATABASE = f"datalake-curated-{ENV}"

EVENT_DATE_PATH = f"s3://{TEMP_BUCKET}/ccaas/event_dates/survey_api_json/"

CURATED_IVR_PATH = f"s3://{CURATED_BUCKET}/ccaas/survey_customer_sat_ivr/"
CURATED_TXN_PATH = f"s3://{CURATED_BUCKET}/ccaas/survey_sms_web_transactional/"
CURATED_REL_PATH = f"s3://{CURATED_BUCKET}/ccaas/survey_sms_web_relational/"

INTENT_MAPPING_PATH = f"s3://{TEMP_BUCKET}/sentiment_analysis/Intent_Mapping/"

FINAL_IVR_STAGING = f"s3://{CURATED_BUCKET}/sentiment_analysis/staging/ivr/"
FINAL_TXN_STAGING = f"s3://{CURATED_BUCKET}/sentiment_analysis/staging/txn/"
FINAL_REL_STAGING = f"s3://{CURATED_BUCKET}/sentiment_analysis/staging/rel/"

FINAL_IVR_PATH = f"s3://{CURATED_BUCKET}/sentiment_analysis/final/ivr/"
FINAL_TXN_PATH = f"s3://{CURATED_BUCKET}/sentiment_analysis/final/transactional/"
FINAL_REL_PATH = f"s3://{CURATED_BUCKET}/sentiment_analysis/final/relational/"

AGG_IVR_FILE = f"s3://{CURATED_BUCKET}/sentiment_analysis/aggregate/ivr.parquet"
AGG_TXN_FILE = f"s3://{CURATED_BUCKET}/sentiment_analysis/aggregate/txn.parquet"
AGG_REL_FILE = f"s3://{CURATED_BUCKET}/sentiment_analysis/aggregate/rel.parquet"

YTD_IVR_FILE = f"s3://{CURATED_BUCKET}/sentiment_analysis/ytd/ivr"
YTD_TXN_FILE = f"s3://{CURATED_BUCKET}/sentiment_analysis/ytd/txn"
YTD_REL_FILE = f"s3://{CURATED_BUCKET}/sentiment_analysis/ytd/rel"

# temp folders for single-file writes
AGG_IVR_TMP = f"s3://{CURATED_BUCKET}/sentiment_analysis/aggregate/tmp_ivr/"
AGG_TXN_TMP = f"s3://{CURATED_BUCKET}/sentiment_analysis/aggregate/tmp_txn/"
AGG_REL_TMP = f"s3://{CURATED_BUCKET}/sentiment_analysis/aggregate/tmp_rel/"

----

enriched = df.join(join_source, on=survey_join_key, how="left")
    return enriched, survey_join_key

def get_latest_partition_value(bucket: str, prefix: str, partition_key: str):
    """
    Lists partition-style 'folders' directly under s3://bucket/prefix that
    follow the partition_key= naming convention (e.g. insert_date=2026-07-08/),
    and returns the latest value found (lexical max — correct for YYYY-MM-DD).
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
------
#Parameter block — add CAS_JOIN_KEY

for opt in ["mode", "ENV", "ARREARS_JOIN_KEY", "CAS_JOIN_KEY"]:
    if f"--{opt}" in sys.argv:
        optional_args.append(opt)

args = getResolvedOptions(sys.argv, required_args + optional_args)
processing_mode = args.get("mode", "manual").lower()
ENV = args.get("ENV", "dev").lower()
ARREARS_JOIN_KEY = args.get("ARREARS_JOIN_KEY", "account_number")
CAS_JOIN_KEY = args.get("CAS_JOIN_KEY", "account_number")
-----

#PATHS block — add CAS bucket/prefix, derived from ENV
#Add this after CURATED_BUCKET is declared:
CAS_BUCKET = CURATED_BUCKET
CAS_PREFIX = "cas/"

------
#New CAS load block — place after your existing arrears load block

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


-----

#Update the three existing arrears call sites to use the renamed function
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
----

#Add CAS call site immediately after the arrears call site
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