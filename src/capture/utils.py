from __future__ import annotations
import hashlib, gzip, io, os, requests, datetime as dt
from dataclasses import dataclass
from typing import Optional
from cz_elections_live.utils.logging import get_logger

S3_ENABLED = bool(os.getenv("S3_BUCKET"))
if S3_ENABLED:
    import boto3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-central-1"))

def utcnow_iso():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def http_get(url: str, timeout: int = 20) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def gz_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    return buf.getvalue()

def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def save_local(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def s3_put(bucket: str, key: str, data: bytes):
    if not S3_ENABLED: 
        return
    logger = get_logger("capture.s3")
    logger.debug(f"Uploading to S3: s3://{bucket}/{key} ({len(data)} bytes)")
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    logger.debug(f"S3 upload completed: {key}")

@dataclass
class SavedObject:
    local_path: str
    s3_key: Optional[str]
    size: int
    sha1: str
    url: str
    fetched_at: str  # ISO8601

def store_blob(raw: bytes, local_path: str, s3_key: Optional[str], url: str) -> SavedObject:
    gz = gz_bytes(raw)
    digest = sha1(gz)
    save_local(local_path, gz)
    if S3_ENABLED and s3_key:
        s3_put(os.environ["S3_BUCKET"], s3_key, gz)
    return SavedObject(local_path, s3_key, len(gz), digest, url, utcnow_iso())
