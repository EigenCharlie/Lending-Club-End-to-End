-- Time series forecasts with actuals and conformal intervals
with ts as (
    select * from {{ ref('stg_time_series') }}
),

forecasts as (
    select * from read_parquet('../data/processed/ts_forecasts.parquet')
)

select
    forecasts.*
from forecasts
