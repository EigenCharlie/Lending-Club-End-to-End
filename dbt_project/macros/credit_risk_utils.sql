{% macro risk_bucket(pd_column) %}
    case
        when {{ pd_column }} < 0.05 then 'Low Risk'
        when {{ pd_column }} < 0.15 then 'Medium Risk'
        when {{ pd_column }} < 0.30 then 'High Risk'
        else 'Very High Risk'
    end
{% endmacro %}


{% macro expected_loss(pd_col, lgd_val, ead_col) %}
    {{ pd_col }} * {{ lgd_val }} * {{ ead_col }}
{% endmacro %}
