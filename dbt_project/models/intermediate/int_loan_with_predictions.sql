-- Joins loan master with PD predictions for test-set loans
with loans as (
    select * from {{ ref('stg_loan_master') }}
),

predictions as (
    select * from {{ ref('stg_test_predictions') }}
)

select
    l.*,
    p.pd_logreg,
    p.pd_catboost_default,
    p.pd_catboost_tuned,
    p.pd_calibrated,
    p.y_true
from loans l
inner join predictions p on l.loan_id = p.loan_id
