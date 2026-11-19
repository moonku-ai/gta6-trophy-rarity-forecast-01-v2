"""gta6-trophy-rarity-forecast-01 v2 — cohort-dilution forecaster (pure Python stdlib).

Model (written from scratch for this entry):
  rate_i(t) = E_i(t) / P(t) — earners over players tracked — forecast the two
  separately instead of extrapolating the rate itself.

  * Players P(t): when the history moves, a power law log P = a + b*log(1+t)
    fitted on the reads and extrapolated to the target. When the history is flat
    (launch night: the page had not refreshed), a refresh prior: the page's
    staleness drives a refresh hazard, and the first refresh multiplies the
    tracked players (~6x, the launch-rush pattern).
  * Earners E_i(t): two cohorts. Veterans (tracked at the last read) progress
    slowly toward their trophy, at a speed set by the trophy's time-to-unlock.
    Newcomers (arriving with the refresh / growth) hold each trophy with a
    probability concentrated on the earliest trophies: a newcomer a few hours
    after release holds ~only the intro trophy, so the refresh dilutes every
    other rate hard. The newcomer profile widens as the target moves later
    (newcomers have played longer by then).
  * Time-to-unlock per trophy comes from its description (chapter order, counts,
    grind size) — a prior the trend fit overrides when the history is rich.
  * Quantiles: when a refresh is likely but not certain, the forecast is a
    two-branch mixture (refresh lands / it does not) in logit space and the
    quantiles are solved from the mixture by bisection, so the 5% tail sits on
    the dilution outcome and the 95% tail on the no-refresh outcome.

Scoring: mean pinball loss on the logit scale; the median is the honest centre
and the spread is calibrated so the 90% band covers about 90% on backtest.
Deterministic: no randomness, no wall-clock reads, byte-identical reruns.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = "2026-11-19T00:00:00Z"

# --- tuned constants (round values; tuned on the practice proxy + synthetic
# backtest in the work dir, never on another entry's code) ---
FIRST_REFRESH_MULT = 6.0      # tracked players at the first refresh vs before
REFRESH_HALFLIFE_H = 1.5      # staleness halflife for the refresh hazard
NEWCOMER_U0_H = 0.25          # newcomer effective playtime at release (hours)
NEWCOMER_U_PER_H = 0.02       # ... growing with hours since release
NEWCOMER_MU_PER_U = 0.12      # extra trophies per newcomer per hour of U
VETERAN_PROG_RATE = 0.15      # veteran progression speed factor
RW_SIGMA = 0.5                # random-walk logit sigma per sqrt(day), house-style
SPREAD_SHRINK = 0.6           # the house spread overstates the trend's own error
BASE_SPREAD = 0.15            # floor logit spread (model error)
MAX_SPREAD = 2.5


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JS rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(iso):
    """ISO 8601 UTC -> epoch seconds (stdlib only)."""
    s = iso.replace("Z", "+00:00")
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def hours_since_release(iso):
    return (parse_time(iso) - parse_time(RELEASE)) / 3600.0


# --- time-to-unlock prior, from the trophy's own description ---
# Ordered (predicate, median play-hours to unlock). Specific rules first.
_UNLOCK_RULES = [
    (lambda n, d: "gone fishing" in d, 0.5),
    (lambda n, d: "chapter 1" in d, 2.5),
    (lambda n, d: "chapter 2" in d, 5.0),
    (lambda n, d: "chapter 3" in d, 9.0),
    (lambda n, d: "chapter 4" in d, 14.0),
    (lambda n, d: "final mission" in d and "complete the final mission." in d, 35.0),
    (lambda n, d: "100% completion" in d, 110.0),
    (lambda n, d: "250 stunts" in d, 55.0),
    (lambda n, d: "100 locations" in d, 45.0),
    (lambda n, d: "50 challenges" in d, 42.0),
    (lambda n, d: "collectible card sets" in d, 38.0),
    (lambda n, d: "8 mission strands" in d, 32.0),
    (lambda n, d: "gold medal in all" in d, 28.0),
    (lambda n, d: "win 20 different races" in d, 22.0),
    (lambda n, d: "drone race" in d, 22.0),
    (lambda n, d: "rob 5 bank vaults" in d, 22.0),
    (lambda n, d: "operation savage glory" in d, 22.0),
    (lambda n, d: "final mission for billy" in d, 22.0),
    (lambda n, d: "final mission for randy" in d, 24.0),
    (lambda n, d: "final mission for tyrone" in d, 26.0),
    (lambda n, d: "final mission for kenton" in d, 20.0),
    (lambda n, d: "final mission for andy" in d, 18.0),
    (lambda n, d: "vcia takedown" in d, 18.0),
    (lambda n, d: "meditate at every spot" in d, 18.0),
    (lambda n, d: "explore all of leonida" in d, 18.0),
    (lambda n, d: "spend $100,000" in d, 18.0),
    (lambda n, d: "headliners" in d, 15.0),
    (lambda n, d: "discover 100 vehicles" in d, 12.0),
    (lambda n, d: "child's play" in d, 12.0),
    (lambda n, d: "second time's the charm" in d, 11.0),
    (lambda n, d: "what the hell just happened" in d, 11.0),
    (lambda n, d: "register 10 vehicles" in d, 10.0),
    (lambda n, d: "cartel heist" in d, 9.0),
    (lambda n, d: "partnership with mass appeal" in d, 9.0),
    (lambda n, d: "choices made" in d, 9.0),
    (lambda n, d: "hunt a unique animal" in d, 8.0),
    (lambda n, d: "lose a 4 star wanted level" in d, 7.0),
    (lambda n, d: "unscheduled departure" in d, 7.0),
    (lambda n, d: "jewelry store" in d, 7.0),
    (lambda n, d: "armored truck" in d, 7.0),
    (lambda n, d: "breakfast run" in d, 6.0),
    (lambda n, d: "rob 3 shops" in d, 5.0),
    (lambda n, d: "take down 5 enemies" in d, 5.0),
    (lambda n, d: "melee drive-by" in d, 5.0),
    (lambda n, d: "try each type of activity once" in d, 4.0),
    (lambda n, d: "vehicle showroom" in d, 3.0),
    (lambda n, d: "fishing challenge" in d, 2.5),
    (lambda n, d: "arcade game" in d, 1.5),
    (lambda n, d: "cocktail with a view" in d, 1.5),
    (lambda n, d: "bowling alley" in d, 1.5),
]


def time_to_unlock_hours(trophy):
    """Median play-hours a player needs to unlock this trophy (prior)."""
    name = trophy.get("name", "").lower()
    desc = trophy.get("description", "").lower()
    if trophy.get("tier") == "platinum":
        return 110.0
    for pred, hours in _UNLOCK_RULES:
        if pred(name, desc):
            return hours
    return {"gold": 35.0, "silver": 25.0, "bronze": 12.0}.get(trophy.get("tier"), 12.0)


def _fit_power_law(xs, ys):
    """Least squares log y = a + b*log(1+x). Returns (a, b, resid_sd) or None."""
    pts = [(math.log1p(x), math.log(y)) for x, y in zip(xs, ys) if y and y > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = sxy / sxx
    a = my - b * mx
    resid = [p[1] - (a + b * p[0]) for p in pts]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
    return a, b, sd


def _staleness_hours(case):
    """Hours the page's data has been unchanged at the checkpoint."""
    hist = case["history"]
    if not hist:
        return 0.0
    at = parse_time(case["at"])
    last = hist[-1]
    # last read whose rates/players differ from the final read
    idx = len(hist) - 1
    while idx > 0:
        prev = hist[idx - 1]
        if prev["players_tracked"] != last["players_tracked"] or prev["rates"] != last["rates"]:
            break
        idx -= 1
    return (at - parse_time(hist[idx]["at"])) / 3600.0


def _refresh_probability(stale_h, horizon_h):
    """P(at least one refresh lands before the target)."""
    if horizon_h <= 0:
        return 0.0
    # hazard grows with staleness: a page stale for hours is overdue a refresh
    lam = (1.0 + stale_h / 3.0) / REFRESH_HALFLIFE_H
    p = 1.0 - math.exp(-lam * horizon_h)
    return min(0.9, max(0.05, p))


def _capped_profile(raw_w, flat_w, total):
    """Distribute `total` over trophies in proportion to raw_w, capping each
    share at 1.0 (a newcomer holds a trophy at most once); the excess left
    after a cap is redistributed in proportion to flat_w (the cohort's own
    rates), so it spreads thinly instead of piling onto the next-fastest
    trophies. Deterministic."""
    q = [0.0] * len(raw_w)
    remaining = total
    active = [i for i, w in enumerate(raw_w) if w > 0]
    while active and remaining > 1e-12:
        wsum = sum(raw_w[i] for i in active)
        if wsum <= 0:
            break
        capped = []
        for i in active:
            share = remaining * raw_w[i] / wsum
            if share >= 1.0:
                capped.append(i)
        if not capped:
            for i in active:
                q[i] += remaining * raw_w[i] / wsum
            break
        for i in capped:
            q[i] = 1.0
            remaining -= 1.0
        active = [i for i in active if i not in capped]
        # the excess spreads over what is left in proportion to flat_w
        raw_w = list(flat_w)
    return q


def _mixture_quantile(mu1, sd1, w1, mu2, sd2, tau):
    """Tau-quantile of w1*N(mu1,sd1) + (1-w1)*N(mu2,sd2), by bisection."""
    from statistics import NormalDist
    nd = NormalDist()
    lo = min(mu1 - 6 * sd1, mu2 - 6 * sd2)
    hi = max(mu1 + 6 * sd1, mu2 + 6 * sd2)

    def cdf(x):
        return w1 * nd.cdf((x - mu1) / sd1) + (1 - w1) * nd.cdf((x - mu2) / sd2)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    hist = case["history"]
    if not hist:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    t_last = hours_since_release(hist[-1]["at"])
    t_tgt = hours_since_release(case["target"])
    p_last = hist[-1]["players_tracked"]
    last_rates = hist[-1]["rates"]

    times = [hours_since_release(h["at"]) for h in hist]
    players = [h["players_tracked"] for h in hist]
    players_vary = max(players) > 1.05 * min(players)
    rates_vary = any(
        h["rates"] is not None and h["rates"] != hist[-1]["rates"] for h in hist
    )

    # ---- players at target: two branches (refresh lands / it does not) ----
    stale_h = _staleness_hours(case)
    p_refresh = _refresh_probability(stale_h, horizon_h)
    if players_vary:
        # fit the POWER LAW on the LAST 24h only: the launch rush saturates,
        # so the whole-history slope would extrapolate the early rush forever
        recent = [(t, p) for t, p in zip(times, players) if t >= t_last - 24.0]
        fit = _fit_power_law([x[0] for x in recent], [x[1] for x in recent])
        if fit is not None:
            a, b, _sd = fit
            p_trend = math.exp(a + b * math.log1p(max(0.0, t_tgt)))
            # sanity cap: tracked players cannot multiply without bound in 4.5d
            p_ref = min(max(p_last, p_trend), p_last * 4.0)
            # a refresh is near-certain over a multi-day horizon
            p_refresh = max(p_refresh, 0.9 if horizon_h > 24 else p_refresh)
        else:
            p_ref = p_last
    else:
        # flat history: the only move is a refresh jump
        p_ref = p_last * FIRST_REFRESH_MULT
    p_ref = max(p_last, p_ref)

    # ---- newcomer profile ----
    # Newcomers are a less-progressed version of the tracked cohort: their
    # holding probability for a trophy is anchored to the last read's rate,
    # discounted by the trophy's time-to-unlock (a newcomer a few hours after
    # release holds ~only the intro trophy), and capped at 1.0 per trophy.
    # The total trophies per newcomer (mu) is ~1.0 at the first refresh.
    u_eff = NEWCOMER_U0_H + NEWCOMER_U_PER_H * max(0.0, t_tgt)
    mu_new = 1.0 + NEWCOMER_MU_PER_U * u_eff  # trophies held per newcomer
    unlock = [time_to_unlock_hours(t) for t in trophies]
    raw_w = []
    flat_w = []
    for i, t in enumerate(trophies):
        r = last_rates[i] if last_rates[i] is not None else 0.0
        raw_w.append(max(r, 0.0) * math.exp(-unlock[i] / max(0.05, u_eff)))
        flat_w.append(max(r, 0.0))
    q_new = _capped_profile(raw_w, flat_w, mu_new)

    # ---- per-trophy trend fit (only when the history actually moves) ----
    # Logit-rate straight line on ln(1+hours) over the last 24h, as the house
    # curve baseline does: rates saturate, so a rate trend extrapolates far
    # more safely than an earners power law. Also keep the fit's slope
    # standard error and mean x: the honest spread of a 4.5-day extrapolation
    # is dominated by slope uncertainty, not by the random-walk prior.
    trend_mu = [None] * len(trophies)
    trend_se = [None] * len(trophies)
    trend_mx = [None] * len(trophies)
    if rates_vary:
        for i in range(len(trophies)):
            xs, ys = [], []
            for h, t in zip(hist, times):
                if t < t_last - 24.0:
                    continue
                r = h["rates"][i] if h["rates"] else None
                if r is not None and r > 0:
                    xs.append(math.log1p(max(0.0, t)))
                    ys.append(logit(r))
            if len(xs) >= 4:
                n = len(xs)
                mx = sum(xs) / n
                my = sum(ys) / n
                sxx = sum((x - mx) ** 2 for x in xs)
                if sxx > 0:
                    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                    a = my - b * mx
                    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
                    sd_res = math.sqrt(sum(r * r for r in resid) / max(1, n - 2))
                    se_b = sd_res / math.sqrt(sxx)
                    trend_mu[i] = inv_logit(a + b * math.log1p(max(0.0, t_tgt)))
                    trend_se[i] = se_b
                    trend_mx[i] = mx

    # ---- cohort model + quantiles ----
    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.0
        if r_last <= 0.0:
            # a trophy at 0% in the last read is unmeasured, not impossible:
            # floor it at a small tier-based rate so the model can still move
            r_last = {"platinum": 0.05, "gold": 0.10, "silver": 0.15}.get(
                trophy.get("tier"), 0.30)
        tau = 1.5 * unlock[i]
        e_last = r_last * p_last / 100.0
        # veterans progress a little: share of the remaining pool crossing
        g = VETERAN_PROG_RATE * (horizon_h / max(1.0, tau))
        g = min(0.5, g)
        e_vet = min(float(p_last), e_last * (1.0 + g))
        # refresh branch: newcomers arrive, holding mainly the earliest trophies
        e_new = (p_ref - p_last) * q_new[i]
        e_tgt = min(float(p_ref), e_vet + e_new)
        m_refresh = 100.0 * e_tgt / p_ref
        # no-refresh branch: same earners, players unchanged
        m_norefresh = 100.0 * e_vet / p_last

        if trend_mu[i] is not None:
            # Rich history: the trend fit carries the forecast on its own.
            # (Measured on the synthetic final-regime backtest: blending the
            # cohort median in here cost ~0.25 loss — the cohort model's
            # players forecast is the least certain part, and the trend
            # already prices the dilution it has seen. The cohort model
            # stays the forecaster where the history is flat and no trend
            # exists at all.) Spread: random-walk prior shrunk (the house
            # baseline's spread is far wider than the trend's own error)
            # plus the slope's standard error times the extrapolation gap.
            mu = logit(trend_mu[i])
            slope_term = trend_se[i] * abs(math.log1p(max(0.0, t_tgt)) - trend_mx[i])
            sd_t = math.sqrt((RW_SIGMA * math.sqrt(horizon_days) * SPREAD_SHRINK) ** 2
                             + BASE_SPREAD ** 2 + slope_term ** 2)
            sd_t = min(MAX_SPREAD, max(0.12, sd_t))
            qs = [round4(min(100.0, max(0.0, inv_logit(mu + z * sd_t)))) for z in Z]
            for k in range(1, 5):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
            rows.append({"id": trophy["id"], "q": qs})
            continue

        sd = math.sqrt(RW_SIGMA * RW_SIGMA * horizon_days + BASE_SPREAD ** 2)
        sd = min(MAX_SPREAD, max(0.15, sd))
        sd_branch = max(0.15, 0.6 * sd)

        mu_r = logit(m_refresh)
        mu_n = logit(m_norefresh)
        if 0.15 < p_refresh < 0.85 and mu_r < mu_n - 0.2:
            qs = [_mixture_quantile(mu_r, sd_branch, p_refresh, mu_n, sd_branch, tau_l)
                  for tau_l in LEVELS]
            qs = [inv_logit(q) for q in qs]
        else:
            mu = mu_r if p_refresh >= 0.5 else mu_n
            if p_refresh >= 0.85:
                mu = mu_r
            qs = [inv_logit(mu + z * sd) for z in Z]
        qs = [round4(min(100.0, max(0.0, q))) for q in qs]
        # enforce non-decreasing after rounding
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
