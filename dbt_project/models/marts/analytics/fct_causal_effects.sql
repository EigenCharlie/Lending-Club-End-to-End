-- Causal treatment effects aggregated by segment
with cate as (
    select * from {{ ref('stg_cate_estimates') }}
)

select
    *
from cate
