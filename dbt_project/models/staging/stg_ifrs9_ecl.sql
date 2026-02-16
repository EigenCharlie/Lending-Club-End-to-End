with source as (
    select * from read_parquet('../data/processed/ifrs9_ecl_comparison.parquet')
)

select
    *
from source
