"""
Database schema models used for history retrieval / storage.
"""

from peewee import (
    AutoField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    IntegerField,
    Model,
)

# Database proxy.
DB_PROXY = DatabaseProxy()

class BaseModel(Model):
    class Meta:
        database = DB_PROXY
        schema = "public"


class DataPoint(BaseModel):
    """
    Each time series stored in RDP's TimescaleDB is represented by an
    entry in the data_points table that hold the general information
    such as human-readable identifiers and selected meta-data fields.
    """
    id = AutoField()
    name = CharField(max_length=128)
    location_code = CharField(max_length=128)
    device_id = CharField(max_length=128)
    data_provider = CharField(max_length=128)

    class Meta:
        table_name = "data_points"


class UnitemporalDoubleDetails(BaseModel):
    """
    Time-series data stored in RDP's TimescaleDB is represented by an
    entry in the unitemporal_double_details table that holds the actual
    data values.

    NOTE: Only double-precision floating point values are supported.
    """
    dp_id = IntegerField()
    valid_time = DateTimeField()
    value = FloatField(null=True)

    class Meta:
        table_name = "unitemporal_double_details"
        primary_key = False
