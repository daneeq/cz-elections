import xml.etree.ElementTree as ET

import pandas as pd
import requests

from cz_elections_live.settings import VOLBY2025_TOPLINE

NS = {"ps": "http://www.volby.cz/ps/"}


def fetch_current_totals() -> pd.DataFrame:
    """
    Fetch current election totals from volby.cz 2025 XML endpoint.

    XML structure per 2025 XSD schema:
    VYSLEDKY > CR > STRANA > HODNOTY_STRANA

    Returns:
        DataFrame with columns: party_code, party_name, votes_reported, pct_reported, timestamp
    """
    try:
        r = requests.get(VOLBY2025_TOPLINE, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        rows = []
        # Get overall CR (Czech Republic) results
        cr = root.find(".//ps:CR", NS)
        if cr is not None:
            # Find all parties under CR
            for strana in cr.findall("ps:STRANA", NS):
                party_code = strana.attrib.get("KSTRANA", "")
                party_name = strana.attrib.get("NAZ_STR", f"Party {party_code}")

                # Get vote data from HODNOTY_STRANA child element
                hodnoty = strana.find("ps:HODNOTY_STRANA", NS)
                if hodnoty is not None:
                    votes = hodnoty.attrib.get("HLASY", "0").replace(",", "")
                    pct = hodnoty.attrib.get("PROC_HLASU", "0").replace(",", ".")

                    rows.append({
                        "party_code": party_code,
                        "party_name": party_name,
                        "votes_reported": int(votes) if votes.isdigit() else 0,
                        "pct_reported": float(pct) if pct.replace(".", "").replace("-", "").isdigit() else 0.0,
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=["party_code", "party_name", "votes_reported", "pct_reported", "timestamp"]
            )
        df["timestamp"] = pd.Timestamp.utcnow()
        return df

    except Exception as e:
        print(f"Error fetching live data: {e}")
        return pd.DataFrame(
            columns=["party_code", "party_name", "votes_reported", "pct_reported", "timestamp"]
        )
