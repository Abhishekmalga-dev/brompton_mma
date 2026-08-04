import os
import boto3
import json
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb")

def lambda_handler(event, context):
    table_name = os.environ.get("TRACKER_TABLE_NAME")
    if not table_name:
        raise EnvironmentError("TRACKER_TABLE_NAME environment variable is not set")
    table = dynamodb.Table(table_name)

    # Compute today's date the same way the Curation Glue job does,
    # so the lookup key matches exactly what was written.
    execution_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"DEBUG: Looking up availability for execution_date={execution_date} in table={table_name}")

    try:
        response = table.get_item(Key={"execution_date": execution_date})
    except Exception as e:
        print(f"ERROR: Failed to read from DynamoDB table {table_name}: {e}")
        raise  # let the Step Function's Catch block handle this as a real failure

    item = response.get("Item")

    if not item:
        print(f"INFO: No availability record found for execution_date={execution_date}. Treating all sources as unavailable.")
        return {
            "ivr_available": False,
            "sms_web_relational_available": False,
            "sms_web_transactional_available": False,
            "api_web_transactional_available": False,
            "api_web_relational_available": False,
            "execution_date": execution_date,
            "lookup_status": "NOT_FOUND"
        }

    print(f"INFO: Found availability record for execution_date={execution_date}: {item}")

    return {
        "ivr_available": bool(item.get("ivr_available", False)),
        "sms_web_relational_available": bool(item.get("sms_web_relational_available", False)),
        "sms_web_transactional_available": bool(item.get("sms_web_transactional_available", False)),
        "api_web_transactional_available": bool(item.get("api_web_transactional_available", False)),
        "api_web_relational_available": bool(item.get("api_web_relational_available", False)),
        "execution_date": execution_date,
        "lookup_status": "FOUND"
    }