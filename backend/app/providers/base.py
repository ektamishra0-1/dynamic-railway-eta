from abc import ABC, abstractmethod

from backend.app.schemas import TrainObservation


class RailwayDataProvider(ABC):

    @abstractmethod
    def get_current_observation(
        self,
        train_no: str,
        journey_date: str,
    ) -> TrainObservation:
        raise NotImplementedError

    @abstractmethod
    def get_downstream_stations(
        self,
        train_no: str,
        journey_date: str,
        station_sequence: int,
        horizon: int = 4,
    ) -> list[dict]:
        raise NotImplementedError
