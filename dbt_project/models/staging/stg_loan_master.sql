with source as (
    select * from read_parquet('../data/processed/loan_master.parquet')
)

select
    cast(id as varchar)          as loan_id,
    loan_amnt,
    annual_inc,
    loan_to_income,
    dti,
    rev_utilization,
    num_delinq_2yrs,
    days_since_last_delinq,
    int_rate,
    int_rate_bucket,
    installment,
    term,
    grade,
    sub_grade,
    home_ownership,
    purpose,
    emp_length,
    verification_status,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    fico_range_low,
    fico_range_high,
    credit_history_months,
    default_flag,
    issue_d,
    loan_status
from source
