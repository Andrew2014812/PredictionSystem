# Risk strategy research — FootPredict v6.1 (findings)

Research only; production Risk unchanged. All numbers from `results/*.json`; the PDF is `reports/v6_1/Risk_Strategy_Research.pdf`.

- Pre-registration commit: `e3dd1bd99361a3399d6d3e173aef28329ce80b41`; data snapshot (local_data): `bf7ccd00976713cc9e6b75d7a590e9ce1ec5423b`.
- Chosen on 2023/24 + 2024/25: **Calibrated Risk (бленд с рынком), пол ≥0 %** (challenger: Calibrated Risk (бленд с рынком), пол ≥0 %).

## ROI per strategy (bets with a real price; 95 % bootstrap CI)

| Strategy | 2023/24 | 2024/25 | 2025/26 (info) | 2026/27 (independent) | Δ vs current 2026/27 |
|---|---|---|---|---|---|
| Текущий Risk | -7.2% [-11.0; -3.4] | -5.9% [-9.6; -2.2] | -5.8% [-9.6; -2.1] | -12.1% [-18.6; -5.2] | — |
| Value Risk (EV ≥ 0) | -6.8% [-10.3; -3.5] | -7.6% [-11.1; -4.2] | -9.1% [-12.4; -5.7] | -10.2% [-16.6; -3.6] | [-3.5; +6.9] |
| Пол EV ≥−3 % | -5.6% [-8.9; -2.3] | -7.6% [-11.1; -4.2] | -7.0% [-10.4; -3.7] | -12.9% [-19.0; -6.3] | [-5.4; +3.7] |
| Пол EV ≥−5 % | -5.3% [-8.7; -2.1] | -7.0% [-10.3; -3.5] | -6.9% [-10.4; -3.5] | -12.7% [-19.2; -6.0] | [-5.3; +3.6] |
| Пол EV ≥−8 % | -6.4% [-9.9; -3.0] | -5.2% [-8.7; -1.7] | -6.9% [-10.5; -3.5] | -13.6% [-19.8; -7.2] | [-5.0; +1.7] |
| Max-EV Risk | -10.6% [-14.0; -7.3] | -7.0% [-10.4; -3.5] | -8.5% [-12.1; -5.2] | -4.1% [-11.5; +3.5] | [+0.2; +15.6] |
| Calibrated Risk, пол ≥0 % | -10.2% [-13.6; -7.2] | -5.6% [-8.9; -2.1] | -8.6% [-12.2; -5.3] | -12.8% [-19.0; -6.1] | [-5.8; +4.6] |
| Calibrated Risk (бленд с рынком), пол ≥0 % | -4.1% [-7.4; -0.7] | -5.5% [-8.9; -2.1] | -3.6% [-7.1; -0.1] | -6.9% [-13.8; -0.1] | [-2.2; +12.2] |
| Calibrated Risk, пол ≥−3 % | -10.8% [-14.1; -7.6] | -6.7% [-10.1; -3.3] | -7.9% [-11.3; -4.7] | -12.2% [-18.3; -5.5] | [-5.2; +4.8] |
| Calibrated Risk (бленд с рынком), пол ≥−3 % | -3.8% [-6.5; -1.0] | -6.3% [-9.6; -3.2] | -4.3% [-7.5; -1.0] | -8.6% [-15.1; -2.0] | [-3.2; +10.5] |
| Calibrated Risk, пол ≥−5 % | -10.1% [-13.4; -6.9] | -4.6% [-8.1; -1.3] | -7.3% [-10.8; -4.1] | -11.8% [-18.0; -5.3] | [-4.6; +4.9] |
| Calibrated Risk (бленд с рынком), пол ≥−5 % | -1.8% [-4.5; +0.8] | -5.9% [-9.2; -2.7] | -3.7% [-6.5; -0.8] | -5.8% [-12.4; +0.1] | [-0.6; +13.3] |
| Calibrated Risk, пол ≥−8 % | -8.5% [-12.0; -5.2] | -5.0% [-8.6; -1.6] | -6.9% [-10.3; -3.4] | -14.1% [-20.3; -7.6] | [-6.1; +2.1] |
| Calibrated Risk (бленд с рынком), пол ≥−8 % | -4.3% [-7.4; -1.2] | -4.9% [-8.3; -1.4] | -4.8% [-8.2; -1.5] | -13.4% [-19.6; -7.0] | [-7.4; +4.6] |
| Calibrated Risk, пол нет | -8.9% [-12.7; -5.4] | -4.7% [-8.3; -1.1] | -6.0% [-9.9; -2.3] | -12.5% [-18.9; -5.7] | [-3.1; +2.3] |
| Calibrated Risk (бленд с рынком), пол нет | -5.4% [-9.1; -1.7] | -5.3% [-9.0; -1.8] | -4.4% [-8.1; -0.8] | -8.9% [-15.4; -2.3] | [-2.8; +8.9] |
| Market-specific Risk | -12.0% [-16.0; -7.9] | -5.9% [-9.6; -2.2] | -5.8% [-9.5; -2.1] | -12.3% [-18.6; -5.6] | [-3.3; +2.8] |
| Conservative Risk (2.50–3.50) | -11.0% [-14.3; -7.9] | -7.8% [-11.1; -4.4] | -6.1% [-9.5; -2.6] | -12.2% [-17.8; -6.2] | [-6.5; +6.5] |
| High-odds Risk (4.50–8.00) | -12.3% [-18.6; -6.3] | -11.1% [-17.0; -4.9] | -10.8% [-17.4; -4.6] | -13.5% [-22.6; -5.2] | [-11.6; +8.1] |

Stored website history 2026/27 (unified_history): 2018 priced Risk bets, ROI -11.17 %, average model EV +8.7%.

## Pre-registered criteria (2026/27)

- C1 (ROI higher): True; C2 (paired CI > 0): False [-2.2; +12.2]; C3 (higher in every season): True -> change recommended: **False**.

## Why model EV does not become ROI

- Calibration p 0.12–0.40 (observed / mean p): 2023/24 0.987 (z -2.1), 2024/25 0.978 (z -3.5), 2025/26* 0.978 (z -3.6), 2026/27 0.977 (z -2.0) — small but significant over-estimation.
- Slope of realised profit on model EV (ideal 1): 2023/24 -0.11, 2024/25 -0.13, 2025/26* -0.08, 2026/27 +0.22.
- Outcome ~ a·logit p_model + b·logit q_market: 2023/24 a=-0.10, b=1.16, 2024/25 a=-0.05, b=1.08, 2025/26* a=0.03, b=1.01.
- Favourite–longshot: blind ROI of every priced selection at odds > 6.00: 2023/24 -25.6%, 2024/25 -19.2%, 2025/26* -30.4%; at 1.50–2.00: 2023/24 -3.6%, 2024/25 -4.1%, 2025/26* -5.4%.
- Max-selection Monte-Carlo (truth = de-vigged price, model noise σ=0.2 in logits): model EV of the chosen bet 2023/24 +7.6 % vs true -5.3 %, 2024/25 +7.1 % vs true -5.7 %, 2025/26* +6.7 % vs true -6.3 %.

## Post-hoc (not pre-registered)

- The pre-registered blend used the de-vigged price of a whole-line Asian handicap as P(win | no refund); corrected to P(win) = q·(1 − refund probability) after the 2026/27 run. Corrected blend, floor ≥0 %: 2023/24 -4.6%, 2024/25 -5.3%, 2025/26* -4.0%, 2026/27 -5.9%.
- Max-EV Risk (worse than current in every history season) is best on 2026/27: Δ [+0.2; +15.6]; Bonferroni-adjusted (k = 18) [-5.6; +20.2].

## Recommendation

Keep the current Risk rule in production. Re-test the market-blend Risk (with the corrected whole-line handicap market probability) under a new pre-registration on matches played after it; change production only if the paired 95 % CI of the ROI difference on those new matches lies above zero and the difference is positive on every historical season.
