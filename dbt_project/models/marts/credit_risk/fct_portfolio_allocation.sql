-- Fact table: portfolio optimization results per loan
with alloc as (
    select * from {{ ref('stg_portfolio_allocations') }}
)

select
    *
from alloc
