-- Dimension table: grade-level summary statistics
with loans as (
    select * from {{ ref('stg_loan_master') }}
),

grade_descriptions as (
    select * from {{ ref('grade_descriptions') }}
)

select
    l.grade,
    gd.description,
    gd.risk_level,
    count(*)                                     as n_loans,
    round(avg(l.default_flag), 4)                as default_rate,
    round(avg(l.loan_amnt), 2)                   as avg_loan_amnt,
    round(avg(l.int_rate), 4)                    as avg_int_rate,
    round(avg(l.annual_inc), 2)                  as avg_annual_inc,
    round(avg(l.dti), 4)                         as avg_dti,
    round(avg(l.fico_range_low), 0)              as avg_fico,
    round(sum(l.loan_amnt), 2)                   as total_loan_volume
from loans l
left join grade_descriptions gd on l.grade = gd.grade
group by l.grade, gd.description, gd.risk_level
order by l.grade
