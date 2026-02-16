"""Feast data source definitions.

Points to parquet files exported by dbt (obt_loan_features).
"""

from feast import FileSource

loan_features_source = FileSource(
    name="loan_features_source",
    path="../data/processed/obt_loan_features.parquet",
    timestamp_field="event_timestamp",
    description="All loan origination features from dbt OBT (One Big Table)",
)
