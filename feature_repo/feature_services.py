"""Feast feature service definitions.

Groups feature views into logical services for different inference needs.
"""

from feast import FeatureService
from feature_views import (
    loan_credit_history_fv,
    loan_demographics_fv,
    loan_origination_fv,
)

pd_prediction_service = FeatureService(
    name="pd_prediction_service",
    features=[loan_origination_fv, loan_credit_history_fv],
    description="Features required for PD (Probability of Default) model inference",
)

risk_assessment_service = FeatureService(
    name="risk_assessment_service",
    features=[loan_origination_fv, loan_credit_history_fv, loan_demographics_fv],
    description="Complete feature set for full risk assessment",
)
