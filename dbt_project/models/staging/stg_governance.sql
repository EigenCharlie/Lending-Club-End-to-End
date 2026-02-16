with source as (
    select * from read_parquet('../data/processed/modeva_governance_checks.parquet')
)

select
    *
from source
