with source as (
    select * from read_parquet('../data/processed/portfolio_allocations.parquet')
)

select
    *
from source
