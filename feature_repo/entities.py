"""Feast entity definitions for the lending club risk project."""

from feast import Entity, ValueType

loan = Entity(
    name="loan",
    join_keys=["loan_id"],
    value_type=ValueType.STRING,
    description="A single loan application in the Lending Club portfolio",
)
