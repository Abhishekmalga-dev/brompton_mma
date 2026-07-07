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
