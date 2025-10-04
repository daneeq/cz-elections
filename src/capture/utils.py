from __future__ import annotations
import hashlib, gzip, io, os, requests, datetime as dt, time
from dataclasses import dataclass
from typing import Optional
from cz_elections_live.utils.logging import get_logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

S3_ENABLED = bool(os.getenv("S3_BUCKET"))
if S3_ENABLED:
    import boto3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-central-1"))

def utcnow_iso():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

# Create a session with retry logic and connection pooling
def _create_session():
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; ElectionsDataCapture/1.0)',
        'Accept': 'application/xml, text/xml, */*',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    })
    
    return session

# Global session for connection reuse
_session = None

def get_session():
    global _session
    if _session is None:
        _session = _create_session()
    return _session

def http_get(url: str, timeout: int = None, max_retries: int = None) -> bytes:
    """
    Fetch URL with retry logic and proper error handling.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (defaults to CAPTURE_HTTP_TIMEOUT env var or 30)
        max_retries: Maximum number of retry attempts (defaults to CAPTURE_MAX_RETRIES env var or 3)
        
    Returns:
        Response content as bytes
        
    Raises:
        requests.RequestException: If all retry attempts fail
    """
    # Use environment variables with defaults
    timeout = timeout or int(os.getenv("CAPTURE_HTTP_TIMEOUT", "30"))
    max_retries = max_retries or int(os.getenv("CAPTURE_MAX_RETRIES", "3"))
    
    session = get_session()
    logger = get_logger("capture.http")
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
            
            # Add small delay between attempts to avoid overwhelming server
            if attempt > 0:
                delay = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                logger.debug(f"Waiting {delay}s before retry")
                time.sleep(delay)
            
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            
            logger.debug(f"Successfully fetched {url} ({len(response.content)} bytes)")
            return response.content
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to connect to {url} after {max_retries} attempts")
                raise
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Timeout connecting to {url} after {max_retries} attempts")
                raise
                
        except requests.exceptions.HTTPError as e:
            # Don't retry on HTTP errors (4xx, 5xx) unless it's a server error
            if e.response and e.response.status_code >= 500:
                logger.warning(f"Server error {e.response.status_code} on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    logger.error(f"Server error from {url} after {max_retries} attempts")
                    raise
            else:
                logger.error(f"HTTP error from {url}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            if attempt == max_retries - 1:
                raise

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
