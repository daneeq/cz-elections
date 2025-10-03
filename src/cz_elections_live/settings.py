import os

from dotenv import load_dotenv

load_dotenv()

DATA_MODE = os.getenv("DATA_MODE", "ps2021_incremental")
VOLBY2025_TOPLINE = os.getenv(
    "VOLBY2025_TOPLINE", "https://www.volby.cz/appdata/ps2025/odata/vysledky.xml"
)
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))
MAJORITY = 101
SEATS_TOTAL = 200
THRESHOLDS = {"single": 0.05, "two": 0.07, "three_plus": 0.11}
