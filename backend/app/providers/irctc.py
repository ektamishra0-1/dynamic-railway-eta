from backend.app.providers.base import RailwayDataProvider


class IRCTCProvider(RailwayDataProvider):
    """
    Integration boundary for an authorized
    railway/IRCTC data source.

    Do NOT scrape or fake an IRCTC API here.

    When authorized API credentials/data access
    are available, implement the provider methods
    while keeping the rest of the system unchanged.
    """

    def get_current_observation(
        self,
        train_no: str,
        journey_date: str,
    ):

        raise NotImplementedError(
            "Authorized railway data source "
            "not configured."
        )


    def get_downstream_stations(
        self,
        train_no: str,
        journey_date: str,
        station_sequence: int,
        horizon: int = 4,
    ):

        raise NotImplementedError(
            "Authorized railway data source "
            "not configured."
        )
