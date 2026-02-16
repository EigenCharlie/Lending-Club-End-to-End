with source as (
    select * from read_parquet('../data/processed/cate_estimates.parquet')
)

select
    *
from source
