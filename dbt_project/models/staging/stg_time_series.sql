with source as (
    select * from read_parquet('../data/processed/time_series.parquet')
)

select
    unique_id,
    ds          as date_month,
    y           as default_rate,
    loan_count,
    avg_int_rate,
    total_amt_funded
from source
