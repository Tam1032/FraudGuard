"""Generate a small fake train_transaction.csv with the IEEE-CIS schema.

Only for smoke tests and CI: it lets the whole pipeline run without the Kaggle data.
Results on this data mean nothing; never report them.

    python scripts/make_synthetic.py --rows 60000 --out data/raw/train_transaction.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400


def generate(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cards = max(rows // 20, 50)
    card_addr = rng.integers(100, 540, n_cards)
    card_start = rng.integers(0, 400, n_cards)  # days the account existed before the data starts
    card = rng.integers(0, n_cards, rows)
    dt = np.sort(rng.integers(DAY, 180 * DAY, rows))

    # Fraud comes in bursts on compromised cards: several fast, larger, odd-hour payments.
    fraud = rng.random(rows) < 0.02
    burst = np.where(fraud)[0]
    for i in burst[: len(burst) // 2]:
        k = rng.integers(2, 5)
        j = slice(i + 1, min(i + 1 + k, rows))
        card[j] = card[i]
        dt[j] = dt[i] + np.sort(rng.integers(30, 3000, len(range(*j.indices(rows)))))
        fraud[j] = True
    order = np.argsort(dt, kind="stable")
    dt, card, fraud = dt[order], card[order], fraud[order]

    late = dt > 130 * DAY  # distribution shift in the last part of the period
    amt = np.round(rng.lognormal(3.8, 1.0, rows) * np.where(fraud, 1.5, 1.0) * np.where(late, 1.3, 1.0), 2)
    hour_shift = np.where(fraud & (rng.random(rows) < 0.3), rng.integers(1, 5, rows) * 3600, 0)
    dt = dt - (dt % DAY) + ((dt % DAY + hour_shift) % DAY)

    domains = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"])
    p_email = rng.choice(domains, rows, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    p_email = np.where(late & (rng.random(rows) < 0.15), "protonmail.com", p_email)
    r_email = np.where(fraud & (rng.random(rows) < 0.5), rng.choice(domains, rows), p_email)

    df = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + rows),
        "isFraud": fraud.astype(int),
        "TransactionDT": dt,
        "TransactionAmt": amt,
        "ProductCD": np.where(fraud & (rng.random(rows) < 0.5), "C", rng.choice(list("WHRSC"), rows, p=[.6, .1, .1, .1, .1])),
        "card1": 1000 + card,
        "card2": rng.choice([111.0, 321.0, 555.0, np.nan], rows),
        "card3": 150.0,
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], rows, p=[.6, .3, .05, .05]),
        "card5": rng.choice([226.0, 224.0, 166.0], rows),
        "card6": np.where(fraud & (rng.random(rows) < 0.4), "credit", rng.choice(["debit", "credit"], rows, p=[.75, .25])),
        "addr1": np.where(fraud & (rng.random(rows) < 0.3), rng.integers(100, 540, rows), card_addr[card]).astype(float),
        "addr2": 87.0,
        "dist1": np.where(rng.random(rows) < 0.5, np.nan, rng.exponential(20, rows) * np.where(fraud, 5, 1)),
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
    })
    for i in range(1, 15):
        df[f"C{i}"] = rng.poisson(np.where(fraud & (rng.random(rows) < 0.3), 2, 1), rows).astype(float)
    df["D1"] = np.floor(dt / DAY + card_start[card])
    new_acct = fraud & (rng.random(rows) < 0.3)
    df.loc[new_acct, "D1"] = rng.integers(0, 3, new_acct.sum())
    df["dist2"] = np.where(rng.random(rows) < 0.9, np.nan, rng.exponential(50, rows))
    for d in range(2, 16):
        df[f"D{d}"] = np.where(rng.random(rows) < 0.3, np.nan, rng.integers(0, 500, rows))
    for m in range(1, 10):
        df[f"M{m}"] = rng.choice(["T", "F", None], rows) if m != 4 else rng.choice(["M0", "M1", "M2", None], rows)
    v_cols = {
        f"V{v}": np.where(rng.random(rows) < 0.3, np.nan, rng.poisson(np.where(fraud, 1.5, 1.0), rows)).astype(float)
        for v in range(1, 340)
    }
    return pd.concat([df, pd.DataFrame(v_cols)], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--out", default="data/raw/train_transaction.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df.isFraud.mean():.2%} fraud) to {out}")
