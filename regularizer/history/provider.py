"""Retrieval of raw historic samples from TimescaleDB."""
from datetime import datetime
from peewee import PostgresqlDatabase, Query
from typing import List, Optional

from ..sample import Sample
from .timescaledb import DB_PROXY, UnitemporalDoubleDetails, DataPoint


class HistoryProvider:
    """
    Retrieves historic samples from TimescaleDB.
    """

    def __init__(self, timescale_config: dict):
        self.db_ = PostgresqlDatabase(
            timescale_config['db'],
            host=timescale_config['host'],
            port=timescale_config['port'],
            user=timescale_config['user'],
            password=timescale_config['password'],
        )
        DB_PROXY.initialize(self.db_)

    def get_history(
            self, dp_name: str,
            start: datetime, end: datetime,
            dp_location_code: Optional[str] = None,
            dp_device_id: Optional[str] = None,
            dp_data_provider: Optional[str] = None
        ) -> List[Sample]:
        """
        Retrieve the history of samples for the given datapoint, start, and end.
        """
        try:
            query = self._get_query(
                dp_name, dp_location_code, dp_device_id, dp_data_provider, start, end
            )
            return self._fetch_samples(query)
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve history for dp_name {dp_name}, "
                f"dp_location_code {dp_location_code}, dp_data_provider {dp_data_provider}, "
                f"start {start}, end {end}: {e}"
            )

    def _get_query(
            self,
            dp_name: str,
            dp_location_code: Optional[str] = None,
            dp_device_id: Optional[str] = None,
            dp_data_provider: Optional[str] = None,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
        ) -> Query:
        """
        Create a query for the given datapoint name, location code, and data provider.
        """
        # Add optional conditions to the query.
        dp_conditions = [DataPoint.name == dp_name]
        if dp_location_code is not None:
            dp_conditions.append(DataPoint.location_code == dp_location_code)
        if dp_device_id is not None:
            dp_conditions.append(DataPoint.device_id == dp_device_id)
        if dp_data_provider is not None:
            dp_conditions.append(DataPoint.data_provider == dp_data_provider)

        # Create a subquery for the datapoint.
        dp_subquery = DataPoint.select(DataPoint.id).where(*dp_conditions)

        # Create a query for the unitemporal double details.
        query = UnitemporalDoubleDetails.select(
            UnitemporalDoubleDetails.valid_time,
            UnitemporalDoubleDetails.value,
        ).where(UnitemporalDoubleDetails.dp_id.in_(dp_subquery))

        # Add optional time conditions to the query.
        if start is not None:
            query = query.where(UnitemporalDoubleDetails.valid_time >= start)
        if end is not None:
            query = query.where(UnitemporalDoubleDetails.valid_time <= end)

        return query

    def _fetch_samples(self, query: Query) -> List[Sample]:
        """
        Fetch the samples from the database.
        """
        with self.db_:
            return [
                Sample(timestamp=valid_time, value=value)
                for valid_time, value in query.tuples()
            ]
