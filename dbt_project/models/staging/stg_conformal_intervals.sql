with source as (
    select
        row_number() over () as row_idx,
        *
    from read_parquet('../data/processed/conformal_intervals_mondrian.parquet')
)

select
    row_idx,
    y_true,
    y_pred          as pd_point,
    pd_low_90       as pd_low,
    pd_high_90      as pd_high,
    width_90        as interval_width,
    pd_low_95       as pd_low_95,
    pd_high_95      as pd_high_95,
    width_95        as interval_width_95,
    grade,
    loan_amnt
from source
