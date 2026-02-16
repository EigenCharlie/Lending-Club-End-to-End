-- Enriches test-set loans with conformal intervals and risk categorization
-- Note: conformal intervals are row-aligned with test predictions (same order)
with loans_preds as (
    select
        row_number() over () as row_idx,
        *
    from {{ ref('int_loan_with_predictions') }}
),

conformal as (
    select * from {{ ref('stg_conformal_intervals') }}
)

select
    lp.loan_id,
    lp.loan_amnt,
    lp.annual_inc,
    lp.int_rate,
    lp.dti,
    lp.grade,
    lp.sub_grade,
    lp.term,
    lp.purpose,
    lp.home_ownership,
    lp.default_flag,
    lp.issue_d,
    lp.pd_calibrated,
    lp.pd_logreg,
    c.pd_low,
    c.pd_high,
    c.interval_width,
    c.pd_low_95,
    c.pd_high_95,
    c.interval_width_95,
    case
        when lp.pd_calibrated < 0.05 then 'Low Risk'
        when lp.pd_calibrated < 0.15 then 'Medium Risk'
        when lp.pd_calibrated < 0.30 then 'High Risk'
        else 'Very High Risk'
    end as risk_category,
    case
        when c.interval_width > 0.5 then 'High Uncertainty'
        when c.interval_width > 0.2 then 'Medium Uncertainty'
        else 'Low Uncertainty'
    end as uncertainty_category
from loans_preds lp
left join conformal c on lp.row_idx = c.row_idx
