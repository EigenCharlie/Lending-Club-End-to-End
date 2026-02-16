-- Fact table: IFRS9 ECL by grade with stage comparison
with ecl as (
    select * from {{ ref('stg_ifrs9_ecl') }}
)

select
    "Grade"            as grade,
    "Avg Loan"         as avg_loan,
    "PD_12m"           as pd_12m,
    "PD_lifetime"      as pd_lifetime,
    "ECL_Stage1"       as ecl_stage1,
    "ECL_Stage2"       as ecl_stage2,
    "Stage2/Stage1"    as stage2_over_stage1
from ecl
