import boto3
import pandas as pd
import io
import os
import urllib.parse

s3 = boto3.client('s3')

# --- Source bucket (hardcoded placeholder — replace with your actual bucket name) ---
INPUT_BUCKET = "xx"

# --- Destination settings (set these as Lambda environment variables) ---
# OUTPUT_BUCKET: the bucket the CSV should be written to.
#   If left blank, the CSV is written back to the SAME bucket the parquet came from.
# OUTPUT_PREFIX: a folder/prefix to prepend to the output key, e.g. "converted/"
#   Leave blank to mirror the same key path as the source file.
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "")


def lambda_handler(event, context):
    """
    Triggered by an S3 'PUT' event when a .parquet file is uploaded to the
    INPUT bucket. Converts it to CSV and writes the result to the OUTPUT
    bucket/prefix configured via environment variables.

    Can also be invoked directly (e.g. via console test event or another
    Lambda) with a payload like:
        { "bucket": "my-input-bucket", "key": "path/to/file.parquet" }
    """

    # --- Figure out source bucket/key whether this came from S3 event or direct invoke ---
    if "Records" in event:
        record = event["Records"][0]
        source_bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
    else:
        source_bucket = event["bucket"]
        key = event["key"]

    if not key.lower().endswith(".parquet"):
        print(f"Skipping non-parquet file: {key}")
        return {"statusCode": 200, "body": f"Skipped: {key} is not a .parquet file"}

    # Safety check: only process files coming from the expected input bucket
    if source_bucket != INPUT_BUCKET:
        print(f"Skipping: {source_bucket} does not match expected INPUT_BUCKET ({INPUT_BUCKET})")
        return {
            "statusCode": 200,
            "body": f"Skipped: event came from '{source_bucket}', expected '{INPUT_BUCKET}'"
        }

    # Fall back to the same bucket if OUTPUT_BUCKET isn't set
    dest_bucket = OUTPUT_BUCKET if OUTPUT_BUCKET else source_bucket

    print(f"Processing s3://{source_bucket}/{key}")

    # --- Download the parquet file into memory ---
    response = s3.get_object(Bucket=source_bucket, Key=key)
    parquet_bytes = response["Body"].read()

    # --- Read parquet into a DataFrame ---
    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    print(f"Read {len(df)} rows, {len(df.columns)} columns")

    # --- Convert to CSV in memory ---
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # --- Build the destination key (same name, .csv extension, optional prefix) ---
    base_key = key.rsplit(".", 1)[0]  # strip ".parquet"
    csv_key = f"{OUTPUT_PREFIX}{base_key}.csv" if OUTPUT_PREFIX else f"{base_key}.csv"

    # --- Upload CSV to the destination bucket ---
    s3.put_object(
        Bucket=dest_bucket,
        Key=csv_key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
    )

    print(f"Wrote CSV to s3://{dest_bucket}/{csv_key}")

    return {
        "statusCode": 200,
        "body": f"Converted s3://{source_bucket}/{key} -> s3://{dest_bucket}/{csv_key}"
    }