-- Fact table: one row per test-set loan with full risk assessment
with risk_profile as (
    select * from {{ ref('int_loan_risk_profile') }}
)

select
    loan_id,
    loan_amnt,
    annual_inc,
    int_rate,
    dti,
    grade,
    sub_grade,
    term,
    purpose,
    home_ownership,
    default_flag,
    issue_d,

    -- PD estimates
    pd_calibrated,
    pd_logreg,

    -- Conformal intervals
    pd_low,
    pd_high,
    interval_width,

    -- Risk classification
    risk_category,
    uncertainty_category,

    -- LGD/EAD proxies
    0.45 as lgd_estimate,
    loan_amnt * 0.80 as ead_estimate,

    -- Expected loss
    pd_calibrated * 0.45 * loan_amnt * 0.80 as expected_loss_point,
    pd_high * 0.45 * loan_amnt * 0.80       as expected_loss_conservative

from risk_profile
