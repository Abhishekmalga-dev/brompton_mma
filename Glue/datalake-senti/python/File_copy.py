"""
Glue Job: copy_curated_prefixes_spark.py
Job Type : Spark (PySpark)

PURPOSE
-------
Copies every object under one or more S3 prefixes from the PROD curated
bucket into the equivalent prefixes in a NON-PROD curated bucket.
Prefixes are supplied as a single comma-separated job parameter.

Parallelism model (two layers, stacked):
  1. NODE level  - the full key list is distributed across Spark
                    partitions/executors via sc.parallelize().
  2. THREAD level - inside each partition, a ThreadPoolExecutor fires
                    many concurrent S3 copy calls from that one executor.

This is what actually justifies using Spark for an I/O-bound copy job
instead of a single Python Shell instance: real multi-node concurrency,
not just multi-threading on one box.

JOB PARAMETERS  (Glue Console -> Job -> Job parameters, or CloudFormation
                 DefaultArguments)
-----------------------------------------------------------------
--JOB_NAME           REQUIRED. Passed automatically by Glue.
--SOURCE_BUCKET      REQUIRED. e.g. psegli-datalakeli-datalake-curated-prod
--TARGET_BUCKET      REQUIRED. e.g. psegli-datalakenonprodli-datalake-curated-dev
--PREFIXES           REQUIRED. Comma separated, e.g.
                      "survey/curated/ivr/,survey/curated/web_txn/,survey/curated/email/"
--NUM_PARTITIONS      OPTIONAL. How many Spark partitions to split the key
                      list into. Default "200". Rule of thumb: roughly
                      2-4x the total number of cores across your worker
                      fleet, or (total_object_count / desired_batch_size)
                      -- whichever gives more partitions, so no single
                      executor is stuck with a huge chunk while others
                      idle.
--THREADS_PER_PARTITION  OPTIONAL. Threads used per executor/partition
                      for concurrent copies. Default "10".
--DRY_RUN             OPTIONAL. "true" or "false". Default "false".
--SSE_KMS_KEY_ID      OPTIONAL. KMS key ARN/ID to encrypt target objects
                      with. If omitted, target bucket default encryption
                      applies.
--WRITE_FAILURE_REPORT OPTIONAL. "true"/"false". Default "true". If any
                      copies fail, writes a small CSV of failed keys +
                      errors to TARGET_BUCKET under
                      _copy_job_reports/<JOB_NAME>/<run_id>/ for audit.

IAM PERMISSIONS the Glue job role needs (attach to the EXISTING shared
role -- do not create a new one, per org policy):
  Source bucket : s3:ListBucket, s3:GetObject  (+ kms:Decrypt if source is
                  SSE-KMS encrypted)
  Target bucket : s3:PutObject, s3:PutObjectAcl (+ kms:GenerateDataKey and
                  kms:Decrypt on the target KMS key if using SSE-KMS)

WORKER SIZING NOTE
-------------------
This is I/O-bound, not CPU/memory-bound, so there is little reason to pay
for G.2X workers. Start with WorkerType=G.1X and NumberOfWorkers in the
5-10 range, then tune NUM_PARTITIONS / THREADS_PER_PARTITION against
observed throughput before scaling workers up further.
"""

import sys
import uuid
import logging
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

# ---------------------------------------------------------------------------
# 1. Resolve job parameters
#    Same pattern as the Python Shell version: required args resolved
#    strictly, optional args resolved only if present, else defaulted.
# ---------------------------------------------------------------------------
REQUIRED_ARGS = ["JOB_NAME", "SOURCE_BUCKET", "TARGET_BUCKET", "PREFIXES"]
OPTIONAL_ARGS = {
    "NUM_PARTITIONS": "200",
    "THREADS_PER_PARTITION": "10",
    "DRY_RUN": "false",
    "SSE_KMS_KEY_ID": "",
    "WRITE_FAILURE_REPORT": "true",
}

args = getResolvedOptions(sys.argv, REQUIRED_ARGS)

passed_keys = [a[2:] for a in sys.argv if a.startswith("--")]
for opt_key, default_val in OPTIONAL_ARGS.items():
    if opt_key in passed_keys:
        args.update(getResolvedOptions(sys.argv, [opt_key]))
    else:
        args[opt_key] = default_val

SOURCE_BUCKET = args["SOURCE_BUCKET"]
TARGET_BUCKET = args["TARGET_BUCKET"]
PREFIXES = [p.strip() for p in args["PREFIXES"].split(",") if p.strip()]
NUM_PARTITIONS = int(args["NUM_PARTITIONS"])
THREADS_PER_PARTITION = int(args["THREADS_PER_PARTITION"])
DRY_RUN = args["DRY_RUN"].strip().lower() == "true"
SSE_KMS_KEY_ID = args["SSE_KMS_KEY_ID"].strip()
WRITE_FAILURE_REPORT = args["WRITE_FAILURE_REPORT"].strip().lower() == "true"

RUN_ID = str(uuid.uuid4())[:8]

# ---------------------------------------------------------------------------
# 2. Glue / Spark boilerplate
#    job.init() / job.commit() give you job-run bookmarking hooks and, more
#    importantly here, correct SUCCEEDED/FAILED status reporting in the
#    Glue console and to anything (Step Functions, EventBridge) watching
#    this job's run state.
# ---------------------------------------------------------------------------
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("copy_curated_prefixes_spark")

s3_client = boto3.client("s3")


# ---------------------------------------------------------------------------
# 3. Driver-side listing
#    Runs once, sequentially, per prefix. This does NOT need to be
#    distributed -- only the copy work benefits from that.
# ---------------------------------------------------------------------------
def list_objects_under_prefix(bucket, prefix):
    """
    Yields (key, size) for every object under a prefix using the
    paginator. list_objects_v2 caps at 1000 keys per call -- skipping
    the paginator would silently drop anything past the first page.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def build_key_list(prefixes):
    all_items = []
    per_prefix_counts = {}
    for prefix in prefixes:
        logger.info(f"Listing objects under s3://{SOURCE_BUCKET}/{prefix}")
        items = list(list_objects_under_prefix(SOURCE_BUCKET, prefix))
        per_prefix_counts[prefix] = len(items)
        if not items:
            logger.warning(f"No objects found under prefix '{prefix}' - skipping.")
        all_items.extend(items)
    return all_items, per_prefix_counts


# ---------------------------------------------------------------------------
# 4. Executor-side copy logic
#    This function runs ONCE PER PARTITION on an executor. It creates a
#    single boto3 client for the whole partition (not per object -- client
#    creation is comparatively expensive), then fires copies concurrently
#    via a thread pool since each copy call is I/O-bound (waiting on S3),
#    not CPU-bound.
# ---------------------------------------------------------------------------
def copy_partition(iterator):
    # A fresh boto3 client per partition. boto3 clients are NOT picklable,
    # so this must be created here (executor side), never at module/driver
    # scope where Spark would try to serialize it into the closure.
    partition_s3 = boto3.client("s3")
    partition_s3_resource = boto3.resource("s3")

    items = list(iterator)
    results = []

    def copy_one(key, size):
        copy_source = {"Bucket": SOURCE_BUCKET, "Key": key}
        extra_args = {"MetadataDirective": "COPY"}
        if SSE_KMS_KEY_ID:
            extra_args["ServerSideEncryption"] = "aws:kms"
            extra_args["SSEKMSKeyId"] = SSE_KMS_KEY_ID

        if DRY_RUN:
            return (key, True, None)

        try:
            # High level .copy() (TransferManager-backed) instead of the
            # low level copy_object API -- copy_object caps out at 5 GB
            # per call, .copy() auto-switches to multipart copy for large
            # objects and has its own retry handling.
            partition_s3_resource.meta.client.copy(
                CopySource=copy_source,
                Bucket=TARGET_BUCKET,
                Key=key,
                ExtraArgs=extra_args,
            )
            return (key, True, None)
        except ClientError as e:
            return (key, False, str(e))

    if not items:
        return iter(results)

    with ThreadPoolExecutor(max_workers=THREADS_PER_PARTITION) as executor:
        futures = {executor.submit(copy_one, k, s): k for k, s in items}
        for future in as_completed(futures):
            results.append(future.result())

    return iter(results)


def main():
    logger.info("=" * 70)
    logger.info("Starting curated data copy job (PROD -> NON-PROD) [Spark]")
    logger.info(f"Source bucket          : {SOURCE_BUCKET}")
    logger.info(f"Target bucket          : {TARGET_BUCKET}")
    logger.info(f"Prefixes               : {PREFIXES}")
    logger.info(f"Num partitions         : {NUM_PARTITIONS}")
    logger.info(f"Threads per partition  : {THREADS_PER_PARTITION}")
    logger.info(f"Dry run                : {DRY_RUN}")
    logger.info(f"Run id                 : {RUN_ID}")
    logger.info("=" * 70)

    all_items, per_prefix_counts = build_key_list(PREFIXES)
    total_objects = len(all_items)

    if total_objects == 0:
        logger.warning("No objects found under any supplied prefix. Nothing to copy.")
        job.commit()
        return

    logger.info(f"Total objects to copy across all prefixes: {total_objects}")

    # Cap partitions at total_objects so we never create more empty
    # partitions than we have work items.
    effective_partitions = min(NUM_PARTITIONS, total_objects)
    keys_rdd = sc.parallelize(all_items, numSlices=effective_partitions)

    results_rdd = keys_rdd.mapPartitions(copy_partition)
    results_rdd.persist()

    success_count = results_rdd.filter(lambda r: r[1]).count()
    # Only collect the FAILED subset back to the driver -- expected to be
    # a small fraction of total_objects, so this is safe even at scale.
    failed_items = results_rdd.filter(lambda r: not r[1]).collect()
    failed_count = len(failed_items)

    logger.info("=" * 70)
    logger.info("JOB SUMMARY")
    logger.info(f"  Per-prefix object counts: {per_prefix_counts}")
    logger.info(f"  TOTAL: {success_count}/{total_objects} succeeded, {failed_count} failed")
    logger.info("=" * 70)

    if failed_count > 0:
        for key, ok, err in failed_items[:50]:  # cap driver-side log spam
            logger.error(f"FAILED: {key} | {err}")
        if failed_count > 50:
            logger.error(f"... plus {failed_count - 50} more failures (see full report).")

        if WRITE_FAILURE_REPORT:
            report_path = (
                f"s3://{TARGET_BUCKET}/_copy_job_reports/"
                f"{args['JOB_NAME']}/{RUN_ID}/"
            )
            failure_df = spark.createDataFrame(
                [(k, e) for k, ok, e in failed_items],
                schema=["failed_key", "error"],
            )
            failure_df.coalesce(1).write.mode("overwrite").option(
                "header", "true"
            ).csv(report_path)
            logger.error(f"Failure report written to {report_path}")

        # Raise (rather than sys.exit) so the Spark driver process ends
        # with a clean stack trace AND job.commit() is never reached --
        # an uncommitted Glue job run correctly shows FAILED status.
        raise RuntimeError(
            f"Copy job completed with {failed_count} failed object copies "
            f"out of {total_objects}."
        )

    logger.info("Job completed successfully - all objects copied.")
    job.commit()


if __name__ == "__main__":
    main()