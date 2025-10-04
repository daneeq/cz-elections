from __future__ import annotations
import os, time, pandas as pd
from .utils import http_get, store_blob
from cz_elections_live.utils.logging import get_logger

STATEFUL_FEEDS = {
    "vysledky": "https://www.volby.cz/appdata/ps2025/odata/vysledky.xml",
    "krajmesta": "https://www.volby.cz/appdata/ps2025/odata/vysledky_krajmesta.xml",
    "zahranici": "https://www.volby.cz/appdata/ps2025/odata/vysledky_zahranici.xml",
    "kandid": "https://www.volby.cz/appdata/ps2025/odata/vysledky_kandid.xml",
}

INTERVAL = int(os.getenv("CAPTURE_STATEFUL_INTERVAL", "60"))
POLL_GRACE = int(os.getenv("CAPTURE_POLL_GRACE", "5"))

def run_stateful_loop(root="data/archive/stateful", index_dir="data/index"):
    logger = get_logger("capture.stateful")
    logger.info(f"Starting stateful capture loop - interval: {INTERVAL}s, grace: {POLL_GRACE}s")
    logger.info(f"Root directory: {root}")
    logger.info(f"Monitoring {len(STATEFUL_FEEDS)} feeds: {list(STATEFUL_FEEDS.keys())}")
    
    idx_rows = []
    while True:
        start = time.time()
        logger.debug("Starting new capture cycle")
        
        for name, url in STATEFUL_FEEDS.items():
            try:
                logger.debug(f"Fetching {name} from {url}")
                raw = http_get(url)
                ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
                local_path = f"{root}/{name}/{ts}.xml.gz"
                s3_prefix = os.getenv("S3_PREFIX", "")
                s3_key = f"{s3_prefix}stateful/{name}/{ts}.xml.gz" if os.getenv("S3_BUCKET") else None
                saved = store_blob(raw, local_path, s3_key, url)
                
                logger.info(f"Captured {name}: {len(raw)} bytes -> {local_path} ({saved.size} bytes compressed)")
                if saved.s3_key:
                    logger.debug(f"Also uploaded to S3: {saved.s3_key}")
                
                idx_rows.append({
                    "kind":"stateful", "name": name, "ts": ts, "url": url,
                    "local_path": saved.local_path, "s3_key": saved.s3_key,
                    "size": saved.size, "sha1": saved.sha1, "fetched_at": saved.fetched_at
                })
            except Exception as e:
                logger.error(f"Failed to capture {name}: {e}")
                idx_rows.append({
                    "kind":"stateful", "name": name, "error": str(e),
                    "ts": pd.Timestamp.utcnow().isoformat()
                })
        
        if idx_rows:
            df = pd.DataFrame(idx_rows)
            os.makedirs(index_dir, exist_ok=True)
            manifest_path = f"{index_dir}/stateful_manifest.parquet"
            df.to_parquet(manifest_path, index=False)
            logger.debug(f"Updated manifest: {manifest_path} ({len(df)} records)")
        
        elapsed = time.time() - start
        sleep_time = max(1, INTERVAL - int(elapsed) - POLL_GRACE)
        logger.debug(f"Cycle completed in {elapsed:.1f}s, sleeping for {sleep_time}s")
        time.sleep(sleep_time)
