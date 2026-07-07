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