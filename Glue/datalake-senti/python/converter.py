import boto3
import pandas as pd
import io
import os
import urllib.parse

s3 = boto3.client('s3')

# Optional: control where the CSV goes relative to the source key.
# Set to "" to place the CSV in the same "folder" as the source file.
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "")


def lambda_handler(event, context):
    """
    Triggered by an S3 'PUT' event when a .parquet file is uploaded.
    Converts it to CSV and writes the result back to the same bucket.

    Can also be invoked directly (e.g. via console test event or another
    Lambda) with a payload like:
        { "bucket": "my-bucket", "key": "path/to/file.parquet" }
    """

    # --- Figure out bucket/key whether this came from S3 event or direct invoke ---
    if "Records" in event:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
    else:
        bucket = event["bucket"]
        key = event["key"]

    if not key.lower().endswith(".parquet"):
        print(f"Skipping non-parquet file: {key}")
        return {"statusCode": 200, "body": f"Skipped: {key} is not a .parquet file"}

    print(f"Processing s3://{bucket}/{key}")

    # --- Download the parquet file into memory ---
    response = s3.get_object(Bucket=bucket, Key=key)
    parquet_bytes = response["Body"].read()

    # --- Read parquet into a DataFrame ---
    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    print(f"Read {len(df)} rows, {len(df.columns)} columns")

    # --- Convert to CSV in memory ---
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # --- Build the destination key (same name, .csv extension) ---
    base_key = key.rsplit(".", 1)[0]  # strip ".parquet"
    csv_key = f"{OUTPUT_PREFIX}{base_key}.csv" if OUTPUT_PREFIX else f"{base_key}.csv"

    # --- Upload CSV back to the same bucket ---
    s3.put_object(
        Bucket=bucket,
        Key=csv_key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
    )

    print(f"Wrote CSV to s3://{bucket}/{csv_key}")

    return {
        "statusCode": 200,
        "body": f"Converted s3://{bucket}/{key} -> s3://{bucket}/{csv_key}"
    }