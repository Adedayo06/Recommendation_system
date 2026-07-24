# src/s3_data.py
import pathlib
import boto3

BUCKET = "newrecsys-bucket"
PREFIX = "data"          # matches the keys in your bucket: data/raw/..., data/processed/...
LOCAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"

def fetch(key: str) -> str:
    """key is relative to data/, e.g. 'raw/events.csv'.
    Downloads from S3 on first use, caches locally, returns the local path."""
    local_path = LOCAL_ROOT / key
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(BUCKET, f"{PREFIX}/{key}", str(local_path))
    return str(local_path)