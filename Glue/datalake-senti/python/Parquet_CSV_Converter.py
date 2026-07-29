"""
glue_parquet_to_csv.py

PURPOSE
-------
AWS Glue PySpark job that reads a single Parquet file (or a Parquet
folder) from S3, converts it to CSV, and writes the result to an exact
destination S3 key.

WHY GLUE INSTEAD OF LAMBDA FOR THIS:
Spark has native built-in Parquet and CSV read/write support - no pandas,
no pyarrow, nothing extra to package or compile. This eliminates the
dependency-packaging problem entirely (no Docker image, no Lambda layer,
no glibc compatibility risk). The tradeoff is Glue's per-run cost is
higher than Lambda's (billed per DPU-second with a startup overhead of
roughly 1-2 minutes for the Spark cluster to spin up), but for a script
that will be maintained long-term alongside your other Glue jobs, that
tradeoff is usually worth it for the operational simplicity.

WHY coalesce(1) + boto3 RENAME:
Spark writes one output file PER PARTITION by default, into a folder,
with an auto-generated filename (e.g. part-00000-<uuid>.csv). That's
standard/expected behavior for distributed writes, but it doesn't match
"put a CSV file at this exact S3 path," which is what was asked for.
coalesce(1) forces all data through a single partition before the write,
producing exactly one part-file. We then use boto3 to copy that one file
to the exact destination key and clean up the temporary folder.

IMPORTANT CAVEAT: coalesce(1) means the final write step runs on a single
executor - you lose write parallelism. Fine for small-to-medium files
(the kind a single Parquet file conversion usually is). If you're ever
converting something in the multi-GB range, drop the coalesce(1) and
accept the partitioned-folder output instead - see the table in my
explanation for when to choose which.

JOB PARAMETERS (passed in via --arguments when starting the job run)
----------------------------------------------------------------------
--source_bucket   S3 bucket containing the source Parquet file
--source_key      Full key (path) to the source Parquet file
--dest_bucket     S3 bucket to write the CSV to
--dest_key        Full key (path + filename) for the final CSV file
"""

import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# ---------------------------------------------------------------------
# STEP 1: Resolve job parameters.
# getResolvedOptions() is the standard Glue way to read arguments passed
# into the job run (via --arguments in start-job-run, or hardcoded in the
# Glue console's job parameters). This is the same pattern you're already
# using in datalake-sentiment-analysis-dev.py for ENV/ACCOUNT_TIER/etc.
# ---------------------------------------------------------------------
args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "source_bucket", "source_key", "dest_bucket", "dest_key"],
)

SOURCE_BUCKET = args["source_bucket"]
SOURCE_KEY = args["source_key"]
DEST_BUCKET = args["dest_bucket"]
DEST_KEY = args["dest_key"]

# ---------------------------------------------------------------------
# STEP 2: Standard Glue job boilerplate - sets up the Spark context,
# Glue context (Glue's wrapper around Spark with extra convenience
# methods), and job bookmark tracking.
# ---------------------------------------------------------------------
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

print(f"Starting conversion: s3://{SOURCE_BUCKET}/{SOURCE_KEY} -> s3://{DEST_BUCKET}/{DEST_KEY}")

# ---------------------------------------------------------------------
# STEP 3: Read the Parquet file.
# spark.read.parquet() is Spark's native reader - it understands Parquet's
# columnar format and footer metadata directly, no pandas/pyarrow layer
# needed. This returns a Spark DataFrame (distributed across the cluster's
# executors), not a pandas DataFrame (which lives in one process's memory).
# ---------------------------------------------------------------------
source_path = f"s3://{SOURCE_BUCKET}/{SOURCE_KEY}"
df = spark.read.parquet(source_path)

row_count = df.count()
col_count = len(df.columns)
print(f"Loaded DataFrame with {row_count} rows and {col_count} columns.")

# ---------------------------------------------------------------------
# STEP 4: Write to a TEMPORARY S3 prefix as CSV.
# We write to a temp location first (rather than directly to the final
# destination) because Spark's CSV writer always produces a folder, and
# we don't want that folder structure sitting at your intended final path -
# we'll extract just the one file we want and clean up the temp folder
# in Step 5.
#
# coalesce(1): forces the write into a single partition, so Spark
# produces exactly one part-file instead of one-per-executor. See the
# module docstring above for the tradeoff this introduces.
#
# header=true: writes column names as the first row of the CSV - standard
# expectation for any downstream tool (Excel, pandas, Athena) reading it.
# ---------------------------------------------------------------------
temp_output_path = f"s3://{DEST_BUCKET}/_temp_csv_conversion/{DEST_KEY}"

df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_output_path)
print(f"Wrote temporary CSV output to {temp_output_path}")

# ---------------------------------------------------------------------
# STEP 5: Find the actual part-file Spark created, copy it to the exact
# destination key, then delete the whole temp folder.
# We use boto3 directly here (not a Spark/Glue API) because this is a
# simple S3 file-management operation, not a distributed data operation -
# using Spark for this would be needless overhead.
# ---------------------------------------------------------------------
s3_client = boto3.client("s3")

# Temp path format is "s3://bucket/_temp_csv_conversion/<dest_key>/" -
# strip the "s3://bucket/" prefix to get the actual S3 key prefix to list.
temp_prefix = f"_temp_csv_conversion/{DEST_KEY}/"

response = s3_client.list_objects_v2(Bucket=DEST_BUCKET, Prefix=temp_prefix)
part_file_key = None
for obj in response.get("Contents", []):
    # Spark's actual data file starts with "part-" (e.g. part-00000-...csv).
    # We skip Spark's own bookkeeping files: "_SUCCESS" (a marker file with
    # no data) and anything starting with "." (hidden checksum/crc files).
    if "/part-" in f"/{obj['Key']}" and not obj["Key"].split("/")[-1].startswith("."):
        part_file_key = obj["Key"]
        break

if part_file_key is None:
    raise RuntimeError(
        f"No part-file found under s3://{DEST_BUCKET}/{temp_prefix} - "
        f"the Spark write step may have failed or produced no output."
    )

print(f"Found part-file: s3://{DEST_BUCKET}/{part_file_key}")

# Copy the single part-file to the exact destination key requested.
s3_client.copy_object(
    Bucket=DEST_BUCKET,
    CopySource={"Bucket": DEST_BUCKET, "Key": part_file_key},
    Key=DEST_KEY,
)
print(f"Copied final CSV to s3://{DEST_BUCKET}/{DEST_KEY}")

# Clean up the entire temp folder (part-file, _SUCCESS marker, checksums).
objects_to_delete = [{"Key": obj["Key"]} for obj in response.get("Contents", [])]
if objects_to_delete:
    s3_client.delete_objects(Bucket=DEST_BUCKET, Delete={"Objects": objects_to_delete})
print(f"Cleaned up temporary folder s3://{DEST_BUCKET}/{temp_prefix}")

print(f"Conversion complete: s3://{DEST_BUCKET}/{DEST_KEY} ({row_count} rows, {col_count} columns)")

# ---------------------------------------------------------------------
# STEP 6: Commit the Glue job bookmark. Standard closing call for any
# Glue ETL job - required for job metrics/bookmark tracking to finalize
# correctly.
# ---------------------------------------------------------------------
job.commit()