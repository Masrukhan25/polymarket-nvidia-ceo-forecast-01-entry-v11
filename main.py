import json
import math
import os
import statistics


QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
NORMAL_QUANTILES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
# Per-quantile width multipliers (v11): widen the outer 5%/95% bands, narrow
# the inner 25%/75% bands. The median (z = 0) is unaffected. In logit space
# these widen/narrow the bands asymmetrically around the last price, which is
# what lets low-price markets that spike get a wider upper band.
TAIL_MULTS = (1.3, 0.75, 1.0, 0.75, 1.3)
HORIZON = 240
WINDOW = 360
JUMP_WINDOW = 60
JUMP_THRESHOLD = 0.03
JUMP_MULTIPLIER = 2.0
JUMP_CAP = 0.05
# Logit-space parameters: price changes are modeled in log-odds, not in raw
# price. A 0.01 move at price 0.02 is a much bigger log-odds move than the
# same 0.01 at price 0.50, so the logit transform widens the bands exactly
# where low-price markets spike — and that is where the big pinball losses
# live. Logit diffs are winsorized at +-0.2; the jump threshold is 4x the
# price-space 0.03 (logit scale is ~4x at p = 0.5).
LOGIT_WINSOR = 0.2
LOGIT_EPS = 0.001
SHRINK = 0.7
FALLBACK_SIGMA = 0.0025
SIGMA_MIN = 0.001
SIGMA_MAX = 0.05


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _logit(p):
    p = max(LOGIT_EPS, min(1.0 - LOGIT_EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def forecast_market(market):
    prices = market["prices"]
    if not prices:
        return [[0.5] * len(NORMAL_QUANTILES) for _ in range(HORIZON)]
    last_price = prices[-1]
    lx = _logit(last_price)
    if len(prices) == 1:
        sigma = FALLBACK_SIGMA
    else:
        xs = [_logit(p) for p in prices]
        raw_diffs = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        recent_raw = raw_diffs[-WINDOW:]
        winsorized = [max(-LOGIT_WINSOR, min(LOGIT_WINSOR, d))
                      for d in recent_raw]
        if len(winsorized) < 30:
            sigma = FALLBACK_SIGMA
        else:
            sigma = statistics.stdev(winsorized)
        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
        sigma *= SHRINK
        recent_raw_60 = raw_diffs[-JUMP_WINDOW:]
        # Jump threshold 0.03 in price space becomes 0.12 in logit space
        # (logit scale is ~4x at p = 0.5); a logit jump means a real regime
        # change, not a one-tick wobble.
        if any(abs(d) >= JUMP_THRESHOLD * 4.0 for d in recent_raw_60):
            sigma = min(sigma * JUMP_MULTIPLIER, JUMP_CAP)
    rows = []
    for h in range(1, HORIZON + 1):
        scale = sigma * math.sqrt(h)
        row = [_sigmoid(lx + z * scale * m)
               for z, m in zip(NORMAL_QUANTILES, TAIL_MULTS)]
        row = [max(0.0, min(1.0, v)) for v in row]
        row.sort()
        row = [round(v, 6) for v in row]
        rows.append(row)
    return rows


def process_case(data):
    quantiles = data["quantiles"]
    if quantiles != list(QUANTILE_LEVELS):
        raise ValueError("quantiles list does not match expected levels")
    forecasts = []
    for market in data["markets"]:
        rows = forecast_market(market)
        forecasts.append({"id": market["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".in"):
            continue
        input_path = os.path.join(input_dir, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = process_case(data)
        output_filename = filename[:-3] + ".out"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, sort_keys=True)


if __name__ == "__main__":
    main()
