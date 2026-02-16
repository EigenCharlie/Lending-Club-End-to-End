-- Ensure all PD predictions are valid probabilities [0, 1]
select
    loan_id,
    pd_calibrated
from {{ ref('int_loan_with_predictions') }}
where pd_calibrated < 0 or pd_calibrated > 1
