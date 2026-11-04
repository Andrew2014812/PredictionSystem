"""The "Why …?" card: what drives the probability of one selection, in football terms.

Factors
-------
The factors explain the probability of the selection ITSELF, through the chain of models that
produces it (``chain.explain_market``): a feature's contribution c_j is in probability units and the
contributions of all features add up to P(selection) − P(average match). A factor is one football
quantity; the home and away versions of a statistic ("h_corners_for_form10", "a_corners_for_form10")
form one factor whose evidence shows both teams' values and whose contribution is their sum.

The bar and percentage of a factor are its RELATIVE MODEL INFLUENCE: |c_factor| / Σ |c| over all
factors. Only the most influential factors are shown, so the shown shares need not add up to 100 %,
and a share is never a change of the probability in percentage points.

Exact score shows no factors and no shares: P(2-1) is not monotonic in the expected goals (more
expected goals first raise and then lower it), so "supports" / "works against" is not stable and a
share would suggest a precision that does not exist. The card then explains the chain instead: the
expected goals, the score's probability and the nearest alternatives.

Summary
-------
``chain_note`` is one or two plain sentences on the model outputs the market is computed from
(expected goals, the 1X2 forecast, the corner distribution); it does not repeat the factors or the
recent-form lines.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.names import feature_label
from ..markets.markets import market_group, market_line

_SIDE = re.compile(r"^(?P<side>[ha])_(?P<rest>.+)$")
_PREFIX = re.compile(r"^(Home team|Away team) · ")
NO_SHARE_GROUPS = {"EXACT_SCORE"}
MAX_FACTORS = 5
MIN_SHARE = 0.03          # a supporting factor below 3 % of the influence is shown only to make up three
MIN_AGAINST = 0.10        # the strongest factor against is shown only when it really matters
MIN_RESULT_SHARE = 0.25   # the 1X2 forecast is mentioned for a goal market when it carries this much
INFLUENCE_TIP = ("Only the most influential factors are shown. Percentages represent their share of total "
                 "model influence.")


@dataclass
class CardFactor:
    key: str
    name: str
    evidence: str
    share: float          # relative model influence: |contribution| / Σ |contributions| (0..1)
    supports: bool


def shares_are_meaningful(market: str) -> bool:
    return market_group(market) not in NO_SHARE_GROUPS


def _fmt(feature: str, value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, str):
        return value
    v = float(value)
    name = feature.lower()
    if "rate" in name or "share" in name or name.endswith("_pct"):
        return f"{v * 100:.0f}%" if abs(v) <= 1.0 else f"{v:.0f}%"
    if "position" in name and "norm" not in name:
        return f"{int(round(v))}."
    if "days" in name or name.endswith("_n") or "games" in name or name.endswith("points"):
        return f"{v:.0f}"
    if name.endswith("elo") or "_elo" in name and "diff" not in name:
        return f"{v:.0f}"
    if name.startswith("diff_") or "elo_diff" in name:
        return f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def _group_key(feature: str) -> tuple[str, str | None]:
    m = _SIDE.match(feature)
    return (m.group("rest"), m.group("side")) if m else (feature, None)


def factor_table(explanation: pd.DataFrame) -> pd.DataFrame:
    """Contributions summed per factor (home + away version of a statistic = one factor):
    columns key, contribution, share, features."""
    col = "contribution" if "contribution" in explanation else "shap"
    keys = explanation["feature"].astype(str).map(lambda f: _group_key(f)[0])
    out = (explanation.assign(key=keys).groupby("key", sort=False)
           .agg(contribution=(col, "sum"), features=("feature", list)).reset_index())
    total = float(out["contribution"].abs().sum()) or 1.0
    out["share"] = out["contribution"].abs() / total
    return out.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)


def card_factors(explanation: pd.DataFrame, home: str, away: str, lang: str = "en",
                 limit: int = MAX_FACTORS) -> list[CardFactor]:
    """The most influential factors for the selection in order of influence (3–5), then the strongest
    factor against it when it carries ≥ 10 % of the influence (tagged on the card).
    ``explanation``: columns feature, contribution (or shap), value."""
    if explanation is None or explanation.empty:
        return []
    values = {str(r.feature): r.value for r in explanation.itertuples(index=False)}
    out = []
    for g in factor_table(explanation).itertuples(index=False):
        first = g.features[0]
        name = _PREFIX.sub("", feature_label(first, lang))
        name = name[:1].upper() + name[1:]
        sides = {_group_key(f)[1]: f for f in g.features}
        if g.key == "league":
            from ..ui.labels import league_name
            evidence = league_name(str(values[first]))
        elif "h" in sides or "a" in sides:
            evidence = " · ".join(f"{team}: {_fmt(sides[s], values[sides[s]])}"
                                  for s, team in (("h", home), ("a", away)) if s in sides)
        else:
            evidence = _fmt(first, values[first])
        out.append(CardFactor(g.key, name, evidence, float(g.share), g.contribution > 0))
    # the card answers "why this selection": the most influential factors FOR it (at least three, then
    # only those with a real part of the influence), then the strongest factor against it when it matters
    support = [f for f in out if f.supports]
    against = [f for f in out if not f.supports]
    chosen = [f for i, f in enumerate(support[:limit]) if i < 3 or f.share >= MIN_SHARE]
    if against and against[0].share >= MIN_AGAINST:
        chosen = (chosen[:limit - 1] if len(chosen) >= limit else chosen) + [against[0]]
    return chosen


# ----------------------------------------------------------------------------- the summary note
NOTE = {
    "en": {
        "xg": "{home} {h} and {away} {a} expected goals",
        "total_over_far": "{xg}: about {n} in total, comfortably above the {line} line.",
        "total_over_mid": "{xg}: about {n} in total, above the {line} line.",
        "total_over_near": "{xg}: about {n} in total, just above the {line} line.",
        "total_over_below": "{xg}: about {n} in total, so over {line} needs the game to beat the forecast.",
        "total_under_far": "{xg}: about {n} in total, well below the {line} line.",
        "total_under_mid": "{xg}: about {n} in total, below the {line} line.",
        "total_under_near": "{xg}: about {n} in total, only just below the {line} line.",
        "total_under_above": "{xg}: about {n} in total, so under {line} needs a quieter game than forecast.",
        "result_tight": " A tight game is likely too (draw {d}), and draws are usually low-scoring.",
        "result_open": " A one-sided game is likely too ({team} win {p}), which tends to add goals.",
        "result_against": " The 1X2 forecast ({h} · {d} · {a}) works against it.",
        "btts": "FootPredict gives {home} a {ph} chance to score and {away} {pa} ({xg}).",
        "btts_no": " The likeliest blank is {team}'s ({p} chance of no goal).",
        "hcp_win": "{team} {line} needs {team} to win by {k} or more goals",
        "hcp_win_refund": "{team} {line} needs {team} to win by {k} or more goals; a {r}-goal win refunds the stake",
        "hcp_avoid": "{team} {line} needs {team} to avoid defeat",
        "hcp_half_win": "{team} {line} is simply a {team} win",
        "hcp_avoid_refund": "{team} {line} wins if {team} avoid defeat; a one-goal defeat refunds the stake",
        "hcp_lose_by": "{team} {line} wins unless {team} lose by {k} or more goals",
        "hcp_lose_by_refund": "{team} {line} wins if {team} lose by no more than {k}; a {r}-goal defeat refunds the stake",
        "hcp_result": ". FootPredict has {team} winning {w} and drawing {dr}.",
        "hcp_goals": ". Expected goals {h}–{a} give a goal difference of {d} for {team}.",
        "hcp_push": " Refund chance: {p}.",
        "corners": "FootPredict expects {mu} corners (usually between {lo} and {hi}), {rel} the {line} line.",
        "above": "above", "below": "below", "on": "right on",
        "x12_fav": "{team} are clear favourites: {home} {h} · draw {d} · {away} {a}.",
        "x12_slight": "{team} are slight favourites: {home} {h} · draw {d} · {away} {a}.",
        "x12_open": "An open match: {home} {h} · draw {d} · {away} {a}.",
        "dc": "{sel} covers two of the three results: {r1} {p1} and {r2} {p2}.",
        "draw": "draw", "win": "{team} win",
        "score_top": "{xg} make {score} the likeliest single score; {alts} are close behind.",
        "score": "{xg}; {score} is one of the likelier scores, behind or level with {alts}.",
    },
    "uk": {
        "xg": "очікувані голи: {home} {h}, {away} {a}",
        "total_over_far": "{xg} — разом близько {n}, помітно вище лінії {line}.",
        "total_over_mid": "{xg} — разом близько {n}, вище лінії {line}.",
        "total_over_near": "{xg} — разом близько {n}, трохи вище лінії {line}.",
        "total_over_below": "{xg} — разом близько {n}, тож для більше {line} матч має перевершити прогноз.",
        "total_under_far": "{xg} — разом близько {n}, помітно нижче лінії {line}.",
        "total_under_mid": "{xg} — разом близько {n}, нижче лінії {line}.",
        "total_under_near": "{xg} — разом близько {n}, лише трохи нижче лінії {line}.",
        "total_under_above": "{xg} — разом близько {n}, тож для менше {line} матч має бути спокійнішим за прогноз.",
        "result_tight": " Імовірна й щільна гра (нічия {d}), а нічиї зазвичай малорезультативні.",
        "result_open": " Імовірна й гра в одні ворота (перемога {team} {p}), що зазвичай додає голів.",
        "result_against": " Прогноз 1X2 ({h} · {d} · {a}) працює проти цього.",
        "btts": "FootPredict дає {home} {ph} шансів забити, а {away} — {pa} ({xg}).",
        "btts_no": " Найімовірніше без голу залишиться {team} ({p}).",
        "hcp_win": "{team} {line}: потрібна перемога {team} з різницею {k}+ голи",
        "hcp_win_refund": "{team} {line}: потрібна перемога {team} з різницею {k}+ голи; при перемозі в {r} гол ставка повертається",
        "hcp_avoid": "{team} {line}: {team} не повинні програти",
        "hcp_half_win": "{team} {line} — це просто перемога {team}",
        "hcp_avoid_refund": "{team} {line} грає, якщо {team} не програють; при поразці в один гол ставка повертається",
        "hcp_lose_by": "{team} {line} грає, якщо {team} не програють з різницею {k}+ голи",
        "hcp_lose_by_refund": "{team} {line} грає, якщо {team} програють не більше ніж на {k}; при поразці в {r} голи ставка повертається",
        "hcp_result": ". За FootPredict {team} виграють з імовірністю {w}, нічия — {dr}.",
        "hcp_goals": ". Очікувані голи {h}–{a} дають різницю {d} для {team}.",
        "hcp_push": " Імовірність повернення: {p}.",
        "corners": "FootPredict очікує {mu} кутових (зазвичай від {lo} до {hi}) — {rel} лінії {line}.",
        "above": "вище", "below": "нижче", "on": "якраз на рівні",
        "x12_fav": "{team} — явні фаворити: {home} {h} · нічия {d} · {away} {a}.",
        "x12_slight": "{team} — невеликі фаворити: {home} {h} · нічия {d} · {away} {a}.",
        "x12_open": "Рівний матч: {home} {h} · нічия {d} · {away} {a}.",
        "dc": "{sel} покриває два з трьох результатів: {r1} {p1} і {r2} {p2}.",
        "draw": "нічия", "win": "перемога {team}",
        "score_top": "{xg} — тож {score} найімовірніший рахунок; поруч {alts}.",
        "score": "{xg}; {score} — один з імовірніших рахунків, поруч {alts}.",
    },
}


def _pc(p: float) -> str:
    return f"{p * 100:.0f}%"


def _corner_range(mu: float, nb_size: float | None) -> tuple[int, int]:
    """10th–90th percentile of the corner distribution."""
    from ..markets.distributions import corners_distribution
    cdf = np.cumsum(corners_distribution(mu, nb_size))
    return int(np.searchsorted(cdf, 0.10)), int(np.searchsorted(cdf, 0.90))


def chain_note(market: str, selection: str, values, home: str, away: str, explanation=None,
               lang: str = "en", sel_text: str = "", nb_size: float | None = None) -> str:
    """A short football note on the chain behind the market (``values``: ``chain.ChainValues``)."""
    if values is None:
        return ""
    tx = NOTE.get(lang, NOTE["en"])
    group, line = market_group(market), market_line(market)
    if group == "CORNERS":
        mu = values.exp_corners
        lo, hi = _corner_range(mu, nb_size)
        rel = tx["above"] if mu > line + 0.25 else (tx["below"] if mu < line - 0.25 else tx["on"])
        return tx["corners"].format(mu=f"{mu:.1f}", lo=lo, hi=hi, rel=rel, line=f"{line:g}")
    ph, pd_, pa = values.p1x2
    lh, la = values.xg_home, values.xg_away
    xg = tx["xg"].format(home=home, away=away, h=f"{lh:.2f}", a=f"{la:.2f}")
    if group == "1X2" or group == "DOUBLE_CHANCE" or (group == "HANDICAP" and abs(line) == 0.5):
        if group == "DOUBLE_CHANCE":
            parts = {"1X": (("H", ph), ("D", pd_)), "X2": (("A", pa), ("D", pd_)), "12": (("H", ph), ("A", pa))}
            (c1, p1), (c2, p2) = parts[selection]
            name = lambda c: tx["draw"] if c == "D" else tx["win"].format(team=home if c == "H" else away)  # noqa: E731
            return tx["dc"].format(sel=sel_text, r1=name(c1), p1=_pc(p1), r2=name(c2), p2=_pc(p2))
        if group == "HANDICAP":
            team = home if selection == "H" else away
            shown = line if selection == "H" else -line
            w, dr = (ph, pd_) if selection == "H" else (pa, pd_)
            key = "hcp_half_win" if shown < 0 else "hcp_avoid"
            return (tx[key].format(team=team, line=f"{shown:+g}")
                    + tx["hcp_result"].format(team=team, w=_pc(w), dr=_pc(dr)))
        fav, p = max((("H", ph), ("A", pa)), key=lambda x: x[1])
        team = home if fav == "H" else away
        key = "x12_fav" if p >= 0.55 else ("x12_slight" if p - min(ph, pa) >= 0.10 else "x12_open")
        return tx[key].format(team=team, home=home, away=away, h=_pc(ph), d=_pc(pd_), a=_pc(pa))
    if group == "TOTAL":
        n = lh + la
        gap = n - line
        if selection == "OVER":
            key = ("total_over_far" if gap >= 1.0 else "total_over_mid" if gap >= 0.3 else
                   "total_over_near" if gap >= 0 else "total_over_below")
        else:
            key = ("total_under_far" if gap <= -1.0 else "total_under_mid" if gap <= -0.3 else
                   "total_under_near" if gap <= 0 else "total_under_above")
        text = tx[key].format(xg=xg[:1].upper() + xg[1:], n=f"{n:.1f}", line=f"{line:g}")
        return text + _result_clause(explanation, values, home, away, tx)
    if group == "BTTS":
        text = tx["btts"].format(home=home, away=away, ph=_pc(values.p_home_scores), pa=_pc(values.p_away_scores),
                                 xg=xg)
        if selection == "NO":
            team, p = min(((home, values.p_home_scores), (away, values.p_away_scores)), key=lambda x: x[1])
            text += tx["btts_no"].format(team=team, p=_pc(1 - p))
        return text
    if group == "HANDICAP":
        team = home if selection == "H" else away
        shown = line if selection == "H" else -line
        whole = float(shown).is_integer()
        if shown < 0:
            k = math.floor(-shown) + 1
            text = tx["hcp_win_refund" if whole else "hcp_win"].format(team=team, line=f"{shown:+g}", k=k, r=k - 1)
        elif whole:                                   # +s: wins on a defeat by at most s - 1, refund on s
            m = int(shown) - 1
            text = (tx["hcp_avoid_refund"].format(team=team, line=f"{shown:+g}") if m == 0 else
                    tx["hcp_lose_by_refund"].format(team=team, line=f"{shown:+g}", k=m, r=int(shown)))
        else:                                         # +s.5: loses only on a defeat by ceil(s) or more
            text = tx["hcp_lose_by"].format(team=team, line=f"{shown:+g}", k=math.ceil(shown))
        d = (lh - la) if selection == "H" else (la - lh)
        text += tx["hcp_goals"].format(h=f"{lh:.2f}", a=f"{la:.2f}", d=f"{d:+.2f}", team=team)
        if whole and values.push:
            text += tx["hcp_push"].format(p=_pc(values.push))
        return text
    if group == "EXACT_SCORE":
        score = selection.replace("-", ":")
        alts = ", ".join(f"{s.replace('-', ':')} ({_pc(p)})" for s, p in values.alternatives)
        top = values.alternatives and values.probability >= values.alternatives[0][1] - 1e-12
        key = "score_top" if top else "score"
        return tx[key].format(xg=xg[:1].upper() + xg[1:], score=f"{score} ({_pc(values.probability)})", alts=alts)
    return ""


def _result_clause(explanation, values, home: str, away: str, tx: dict) -> str:
    """For a goal total: mention the 1X2 forecast only when it carries a real part of the influence."""
    if explanation is None:
        return ""
    effect = explanation.model_effect()
    total = float(explanation.output_effect.abs().sum())
    res = float(effect.get("result", 0.0))
    if not total or abs(res) / total < MIN_RESULT_SHARE:
        return ""
    ph, pd_, pa = values.p1x2
    if res > 0:
        fav, p = max(((home, ph), (away, pa)), key=lambda x: x[1])
        # the classifier helps an UNDER through the draw, an OVER through a one-sided game
        return (tx["result_tight"].format(d=_pc(pd_)) if explanation.selection == "UNDER"
                else tx["result_open"].format(team=fav, p=_pc(p)))
    return tx["result_against"].format(h=_pc(ph), d=_pc(pd_), a=_pc(pa))
