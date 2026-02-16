with source as (
    select * from read_parquet('../data/processed/test_predictions.parquet')
)

select
    cast(id as varchar)   as loan_id,
    y_true,
    y_prob_lr             as pd_logreg,
    y_prob_cb_default     as pd_catboost_default,
    y_prob_cb_tuned       as pd_catboost_tuned,
    y_prob_final          as pd_calibrated
from source
