-- Model performance metrics aggregated by issue month cohort
with risk as (
    select * from {{ ref('int_loan_with_predictions') }}
)

select
    date_trunc('month', issue_d) as cohort_month,
    grade,
    count(*)                                     as n_loans,
    round(avg(default_flag), 4)                  as actual_default_rate,
    round(avg(pd_calibrated), 4)                 as avg_predicted_pd,
    round(avg(pd_logreg), 4)                     as avg_predicted_pd_logreg,
    round(avg(abs(pd_calibrated - default_flag)), 4) as mae,
    round(avg(power(pd_calibrated - default_flag, 2)), 6) as mse,
    sum(default_flag)                            as n_defaults
from risk
group by date_trunc('month', issue_d), grade
order by cohort_month, grade
