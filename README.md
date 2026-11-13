# polymarket-nvidia-ceo-forecast-01 — v11: logit-space volatility quantile forecaster

Forecasts each Polymarket "Next CEO of NVIDIA?" market's Yes price for the 240 minutes after each checkpoint, as five quantiles (5%, 25%, 50%, 75%, 95%), optimized for mean pinball loss.

Method (measured, not guessed): work in log-odds (logit) space, not raw price. Persistence median at the last price; per-market sigma from the last 360 one-minute logit differences, winsorized to ±0.2, times 0.7; sigma doubled (capped at 0.05) when any raw logit move in the last 60 minutes reaches 0.12; quantile bands as sigmoid(logit(last price) + normal quantile × sigma × sqrt(horizon minute)), with the outer 5%/95% bands widened x1.3 and the inner 25%/75% bands narrowed x0.75. No drift, no mean reversion, no cross-market normalization — all three made things worse in testing.

Why logit: a 0.01 move at price 0.02 is a much bigger log-odds move than the same 0.01 at price 0.50, so the logit transform widens the bands exactly where low-price markets spike — and that is where the big pinball losses live. In price space the same spike gets a band that is too narrow.

Changes vs v10 (practice 0.274007, held-out 2.329071): the whole model moved to logit space (winsor ±0.2, jump threshold 0.12 in logit units), tail multipliers outer x1.3 / inner x0.75, shrink back to 0.7 and jump window back to 60 minutes (v9 defaults).

Scores measured locally with a scorer that reproduces the house persistence baseline exactly (0.363589 on the six practice cases):
- Practice (6 cases, 153,600 items): **0.300732** (house baseline 0.363589) — worse than v10. Practice does NOT decide the final; the final is 12 fresh sealed cases over 48h.
- Held-out (5 windows cut from earlier, jumpier history, 36,000 items): **2.159582** (house 2.345500; v10 2.329071)
- Broad held-out (34 unique windows cut from the longest case, 429,600 items): **1.261942** (v10 1.305011; v9 1.324407). Logit wins in 21 of 34 windows, wins big on jumpy windows and loses small on calm ones.

Honest caveat: on the 11 most recent windows (closest in time to the final, full market set), v10 is slightly better (0.368571 vs 0.372021). Logit's edge is concentrated in older, jumpier windows. The bet is that over 48h the final will include volatile periods, where logit wins big — but that is a judgment, not a certainty.

Pure Python 3.13 standard library, deterministic, offline, ~0.3–0.7 s per case (limit 8 s).

Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
