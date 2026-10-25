-- Behavioural features for each transaction, computed from the table `tx`.
-- The same query runs at training time (over all history) and in the API
-- (over one client's recent history plus the new transaction), so training
-- and serving features cannot drift apart.
--
-- uid approximates one client: card + billing address + the day the account
-- was first seen (TransactionDT in days minus D1, "days since first transaction").

WITH base AS (
    SELECT
        *,
        concat_ws('_',
            coalesce(CAST(card1 AS VARCHAR), 'na'),
            coalesce(CAST(addr1 AS VARCHAR), 'na'),
            coalesce(CAST(floor(TransactionDT / 86400 - coalesce(D1, 0)) AS VARCHAR), 'na')
        ) AS uid
    FROM tx
)
SELECT
    *,
    ln(1 + TransactionAmt)                                   AS amt_log,
    CAST(floor(TransactionDT / 3600) % 24 AS INTEGER)       AS hour_of_day,
    CAST(floor(TransactionDT / 86400) % 7 AS INTEGER)       AS day_of_week,
    TransactionAmt - floor(TransactionAmt)                  AS amt_cents,
    CAST(P_emaildomain = R_emaildomain AS INTEGER)          AS email_match,

    -- D4/D10/D15 count days since some past event; subtracting them from the
    -- current day gives a date that stays constant for one account.
    floor(TransactionDT / 86400) - D4                       AS D4_start_day,
    floor(TransactionDT / 86400) - D10                      AS D10_start_day,
    floor(TransactionDT / 86400) - D15                      AS D15_start_day,

    -- Velocity: how many transactions this client made recently.
    count(*) OVER w_1h                                      AS uid_tx_1h,
    count(*) OVER w_24h                                     AS uid_tx_24h,
    count(*) OVER w_7d                                      AS uid_tx_7d,
    sum(TransactionAmt) OVER w_24h                          AS uid_amt_sum_24h,

    -- Amount compared with the client's usual spending.
    avg(TransactionAmt) OVER w_30d                          AS uid_amt_mean_30d,
    TransactionAmt / nullif(avg(TransactionAmt) OVER w_30d, 0) AS amt_to_uid_mean_30d,
    (TransactionAmt - avg(TransactionAmt) OVER w_30d)
        / nullif(stddev_samp(TransactionAmt) OVER w_30d, 0) AS amt_zscore_30d,
    stddev_samp(TransactionAmt) OVER w_30d                  AS uid_amt_std_30d,
    avg(dist1) OVER w_30d                                   AS uid_dist1_mean_30d,
    avg(C13) OVER w_30d                                     AS uid_c13_mean_30d,
    avg(D15) OVER w_30d                                     AS uid_d15_mean_30d,

    -- Timing and novelty.
    TransactionDT - lag(TransactionDT) OVER w_all          AS secs_since_prev_tx,
    CAST(ProductCD IS DISTINCT FROM lag(ProductCD) OVER w_all AS INTEGER) AS product_changed,
    CAST(addr1 IS DISTINCT FROM lag(addr1) OVER (PARTITION BY card1 ORDER BY TransactionDT, TransactionID) AS INTEGER)
                                                            AS card_addr_changed
FROM base
WINDOW
    w_1h  AS (PARTITION BY uid ORDER BY TransactionDT RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW),
    w_24h AS (PARTITION BY uid ORDER BY TransactionDT RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW),
    w_7d  AS (PARTITION BY uid ORDER BY TransactionDT RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW),
    w_30d AS (PARTITION BY uid ORDER BY TransactionDT RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW),
    w_all AS (PARTITION BY uid ORDER BY TransactionDT, TransactionID ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
ORDER BY TransactionDT, TransactionID
