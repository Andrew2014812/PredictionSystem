"""Show the odds-API quota used by FootPredict (no API call, no quota spent).

    python scripts/odds_credits.py

OddsPapi: 250 billable requests / month on the free plan (historical odds are free).
The Odds API: 500 credits / month on the free plan.
The numbers come from the committed ledgers, updated from the APIs' own answers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.providers.ledger import CreditLedger  # noqa: E402


def show(title: str, ledger: CreditLedger, unit: str) -> None:
    data = ledger.data
    print(f"== {title}")
    print(f"  {unit} left this month (last known): {data.get('remaining', 'unknown')}"
          f"   (reserve never spent: {ledger.reserve})")
    print(f"  used this month: {data.get('used', 'unknown')} (as of {data.get('checked_at', 'never')})")
    print(f"  spent by FootPredict in {data.get('month', '-')}: {data.get('month_credits', 0)}")
    recent = sorted(data["requests"].items(), key=lambda kv: kv[1]["at"], reverse=True)[:10]
    for key, entry in recent:
        print(f"    {entry['at']}  {entry['cost']:>2} {unit}  {entry['items']:>3} items  "
              f"{entry.get('status', '')}  {key}")


if __name__ == "__main__":
    show("OddsPapi", CreditLedger(config.API_DIR / "oddspapi" / "ledger.json", config.ODDSPAPI_MONTHLY_RESERVE),
         "requests")
    show("The Odds API", CreditLedger(), "credits")
