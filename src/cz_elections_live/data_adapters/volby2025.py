import xml.etree.ElementTree as ET

import pandas as pd
import requests

from cz_elections_live.settings import VOLBY2025_TOPLINE

NS = {"ps": "http://www.volby.cz/ps/"}  # adjust to actual namespace when live


def fetch_current_totals() -> pd.DataFrame:
    r = requests.get(VOLBY2025_TOPLINE, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    # Placeholder paths/attributes; confirm against live XSD
    for s in root.findall(".//ps:STRANA", NS):
        rows.append(
            {
                "party_code": s.attrib.get("KSTRANA"),
                "party_name": s.attrib.get("NAZEV"),
                "votes_reported": int(s.attrib.get("HLASY", "0")),
                "pct_reported": float(str(s.attrib.get("PROCENT", "0")).replace(",", ".")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["party_code", "party_name", "votes_reported", "pct_reported", "timestamp"]
        )
    df["timestamp"] = pd.Timestamp.utcnow()
    return df
