-- Conformal prediction coverage analysis by grade and risk segment
with risk as (
    select * from {{ ref('int_loan_risk_profile') }}
)

select
    grade,
    risk_category,
    uncertainty_category,
    count(*)                                                       as n_loans,
    round(avg(case when default_flag between pd_low and pd_high
              then 1 else 0 end), 4)                               as empirical_coverage,
    round(avg(interval_width), 4)                                  as avg_interval_width,
    round(avg(pd_calibrated), 4)                                   as avg_pd,
    round(avg(pd_low), 4)                                          as avg_pd_low,
    round(avg(pd_high), 4)                                         as avg_pd_high
from risk
where pd_low is not null
group by grade, risk_category, uncertainty_category
order by grade
