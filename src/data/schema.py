import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RawSalesSchema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "store": Column(int, Check.greater_than(0), nullable=False),
        "item": Column(int, Check.greater_than(0), nullable=False),
        "sales": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    },
    strict=False,
    coerce=False,
)


def validate_raw_data(df):
    """Validate raw sales data against the expected schema.

    Raises pandera.errors.SchemaError with a descriptive message on failure.
    """
    return RawSalesSchema.validate(df, lazy=True)
