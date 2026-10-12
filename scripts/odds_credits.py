"""Show The Odds API credit usage recorded by FootPredict (no API call, no credits).

    python scripts/odds_credits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers.ledger import CreditLedger  # noqa: E402

if __name__ == "__main__":
    ledger = CreditLedger()
    data = ledger.data
    print(f"Credits left this month (from the API's last answer): {data.get('remaining', 'unknown')}")
    print(f"Credits used this month: {data.get('used', 'unknown')} (as of {data.get('checked_at', 'never')})")
    print(f"Credits spent by FootPredict in {data.get('month', '-')}: {data.get('month_credits', 0)}")
    recent = sorted(data["requests"].items(), key=lambda kv: kv[1]["at"], reverse=True)[:10]
    for key, entry in recent:
        print(f"  {entry['at']}  {entry['cost']:>2} credits  {entry['items']:>3} priced  {key}")
