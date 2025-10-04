import threading
from .stateful_snapshots import run_stateful_loop
from .batch_snapshots import run_batch_loops
from .config import ARCHIVE_DIR, INDEX_DIR
from cz_elections_live.utils.logging import get_logger

def main():
    logger = get_logger("capture")
    logger.info("Starting capture system...")
    logger.info(f"Archive directory: {ARCHIVE_DIR}")
    logger.info(f"Index directory: {INDEX_DIR}")
    
    # Start stateful capture thread
    logger.info("Starting stateful snapshots thread...")
    t1 = threading.Thread(target=run_stateful_loop, kwargs={"root": f"{ARCHIVE_DIR}/stateful", "index_dir": INDEX_DIR}, daemon=True)
    
    # Start batch capture thread  
    logger.info("Starting batch snapshots thread...")
    t2 = threading.Thread(target=run_batch_loops, kwargs={"root": f"{ARCHIVE_DIR}/batches", "index_dir": INDEX_DIR}, daemon=True)
    
    t1.start(); t2.start()
    logger.info("Both capture threads started successfully. Press Ctrl+C to stop.")
    
    try:
        t1.join(); t2.join()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")

if __name__ == "__main__":
    main()
