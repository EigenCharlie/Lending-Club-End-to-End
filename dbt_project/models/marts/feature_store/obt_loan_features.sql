-- One Big Table: all loan features for Feast feature store export
with loans as (
    select * from {{ ref('stg_loan_master') }}
)

select
    loan_id,
    issue_d                          as event_timestamp,

    -- Core features
    loan_amnt,
    annual_inc,
    loan_to_income,
    dti,
    rev_utilization,
    int_rate,
    installment,
    term,
    grade,
    sub_grade,

    -- Credit history
    num_delinq_2yrs,
    days_since_last_delinq,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    fico_range_low,
    fico_range_high,
    credit_history_months,

    -- Demographics
    home_ownership,
    purpose,
    emp_length,
    verification_status,

    -- Target
    default_flag
from loans
