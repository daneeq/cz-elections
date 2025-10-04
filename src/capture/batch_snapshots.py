from __future__ import annotations
import os, time, pandas as pd, requests
from .utils import http_get, store_blob
from cz_elections_live.utils.logging import get_logger

BATCHES = {
    "okrsky": "https://www.volby.cz/appdata/ps2025/odata/okrsky/vysledky_okrsky_{nn}.xml",
    "okrsky_latest": "https://www.volby.cz/appdata/ps2025/odata/okrsky/vysledky_okrsky.xml",
    "obce": "https://www.volby.cz/appdata/ps2025/odata/obce_d/vysledky_obce_{nn}.xml",
    "obce_latest": "https://www.volby.cz/appdata/ps2025/odata/obce_d/vysledky_obce.xml",
    "okresy": "https://www.volby.cz/appdata/ps2025/odata/okresy_d/vysledky_okresy_{nn}.xml",
    "okresy_latest": "https://www.volby.cz/appdata/ps2025/odata/okresy_d/vysledky_okresy.xml",
}

def format_batch(n: int) -> str:
    return f"{n:05d}"

def run_batch_loops(root="data/archive/batches", index_dir="data/index"):
    logger = get_logger("capture.batch")
    logger.info(f"Starting batch capture loop")
    logger.info(f"Root directory: {root}")
    logger.info(f"Monitoring batch types: okrsky, obce, okresy")
    
    os.makedirs(index_dir, exist_ok=True)
    cursor_file = f"{index_dir}/batch_cursor.txt"
    n = 1
    if os.path.exists(cursor_file):
        try:
            n = int(open(cursor_file).read().strip())
            logger.info(f"Resuming from batch {n} (cursor file found)")
        except:
            logger.info("Starting from batch 1 (invalid cursor file)")
    else:
        logger.info("Starting from batch 1 (no cursor file)")

    manifest_rows = []
    while True:
        nn = format_batch(n)
        logger.debug(f"Checking batch {nn}")
        progressed = False

        for i, kind in enumerate(("okrsky","obce","okresy")):
            # Add delay between requests to avoid overwhelming the server
            if i > 0:
                delay = int(os.getenv("CAPTURE_BATCH_DELAY", "1"))  # Configurable delay between batch requests
                logger.debug(f"Rate limiting: waiting {delay}s before next batch request")
                time.sleep(delay)
                
            url = BATCHES[kind].format(nn=nn)
            try:
                logger.debug(f"Fetching {kind} batch {nn} from {url}")
                raw = http_get(url)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.debug(f"Batch {nn} for {kind} not found (404) - expected for high batch numbers")
                    continue
                logger.error(f"HTTP error fetching {kind} batch {nn}: {e}")
                manifest_rows.append({"kind":kind, "batch": nn, "error": str(e), "ts": pd.Timestamp.utcnow().isoformat()})
                continue

            local_path = f"{root}/{kind}/{nn}.xml.gz"
            s3_prefix = os.getenv("S3_PREFIX", "")
            s3_key = f"{s3_prefix}batches/{kind}/{nn}.xml.gz" if os.getenv("S3_BUCKET") else None
            saved = store_blob(raw, local_path, s3_key, url)
            
            logger.info(f"Captured {kind} batch {nn}: {len(raw)} bytes -> {local_path} ({saved.size} bytes compressed)")
            if saved.s3_key:
                logger.debug(f"Also uploaded to S3: {saved.s3_key}")
            
            manifest_rows.append({
                "kind": kind, "batch": nn, "url": url, "local_path": saved.local_path,
                "s3_key": saved.s3_key, "size": saved.size, "sha1": saved.sha1, "fetched_at": saved.fetched_at
            })
            progressed = True

        if progressed:
            n += 1
            with open(cursor_file, "w") as f:
                f.write(str(n))
            logger.debug(f"Updated cursor to batch {n}")
            if manifest_rows:
                manifest_path = f"{index_dir}/batches_manifest.parquet"
                pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
                logger.debug(f"Updated manifest: {manifest_path} ({len(manifest_rows)} records)")
            continue

        logger.debug("No new batches found, sleeping for 30s")
        time.sleep(30)
