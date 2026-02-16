-- Ensure overall default rate is between 5% and 30% (sanity check)
with stats as (
    select avg(default_flag) as default_rate
    from {{ ref('stg_loan_master') }}
)
select default_rate
from stats
where default_rate < 0.05 or default_rate > 0.30
