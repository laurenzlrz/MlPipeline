from typing import Tuple, Dict, List
from abc import ABC, abstractmethod

import pandas as pd
from matplotlib import pyplot as plt

from processing_pipeline.ICoreElementDoc import IElementDoc
from processing_pipeline.ends.RunFinisher import RunFinisher
from processing_pipeline.stations.InitializingStation import InitializingStation
from processing_pipeline.stations.ProcessStation import ProcessStation
from processing_pipeline.stations.VisualisationStation import VisualisationStation
from processing_pipeline.description_enums import DataOrigin, Column, ProcessType, AbstractionLevel


class RunDataAddProxy(ABC):
    """
    RunDataAddProxy is an abstract base class that defines methods for adding data to a run.
    """

    @abstractmethod
    def add_df(self, key: Tuple[DataOrigin, AbstractionLevel], data: pd.DataFrame):
        """
        Adds a DataFrame to the run.

        :param key: A tuple containing DataOrigin and AbstractionLevel.
        :param data: The DataFrame to add.
        """
        pass

    @abstractmethod
    def add_fig(self, key: Tuple[DataOrigin, AbstractionLevel, Column], fig: plt.plot):
        """
        Adds a figure to the run.

        :param key: A tuple containing DataOrigin, AbstractionLevel, and Column.
        :param fig: The figure to add.
        """
        pass


class RunDataGetProxy(ABC):
    """
    RunDataGetProxy is an abstract base class that defines methods for retrieving data from a run.
    """

    @abstractmethod
    def get_fig(self, key: Tuple[DataOrigin, AbstractionLevel]) -> List[plt.Figure]:
        """
        Retrieves figures associated with the given key.

        :param key: A tuple containing DataOrigin and AbstractionLevel.
        :return: A list of figures associated with the key.
        """
        pass

    @abstractmethod
    def get_df(self, key: Tuple[DataOrigin, AbstractionLevel]) -> List[pd.DataFrame]:
        """
        Retrieves DataFrames associated with the given key.

        :param key: A tuple containing DataOrigin and AbstractionLevel.
        :return: A list of DataFrames associated with the key.
        """
        pass

    @abstractmethod
    def get_df_keys(self) -> List[Tuple[DataOrigin, AbstractionLevel]]:
        """
        Retrieves all DataFrame keys.

        :return: A list of tuples containing DataOrigin and AbstractionLevel.
        """
        pass

    @abstractmethod
    def get_fig_keys(self) -> List[Tuple[DataOrigin, AbstractionLevel, Column]]:
        """
        Retrieves all figure keys.

        :return: A list of tuples containing DataOrigin, AbstractionLevel, and Column.
        """
        pass

    @abstractmethod
    def get_model_adapter(self) -> IElementDoc:
        """
        Retrieves the model adapter.

        :return: The model adapter.
        """
        pass

    @abstractmethod
    def get_module_adapter(self) -> IElementDoc:
        """
        Retrieves the module adapter.

        :return: The module adapter.
        """
        pass

    @abstractmethod
    def get_trainer_adapter(self) -> IElementDoc:
        """
        Retrieves the trainer adapter.

        :return: The trainer adapter.
        """
        pass

    @abstractmethod
    def get_process_station(self):
        """
        Retrieves the process station.

        :return: The process station.
        """
        pass


class Run(RunDataAddProxy, RunDataGetProxy):
    """
    The Run object is the central object that holds all the data and the flow of the run.
    """

    def __init__(self, model_adapter, module_adapter, trainer_adapter,
                 initiation_station: InitializingStation, process_station: ProcessStation,
                 visualisation_station: VisualisationStation, run_finisher: RunFinisher,
                 identifier):
        """
        Initializes the Run with the given adapters, stations, and identifier.

        :param model_adapter: Adapter for the model.
        :param module_adapter: Adapter for the module.
        :param trainer_adapter: Adapter for the trainer.
        :param initiation_station: The station responsible for initialization.
        :param process_station: The station responsible for processing.
        :param visualisation_station: The station responsible for visualization.
        :param run_finisher: The finisher responsible for finalizing the run.
        :param identifier: A unique identifier for the run.
        """
        self._model_adapter = model_adapter
        self._module_adapter = module_adapter
        self._trainer_adapter = trainer_adapter

        self._initiation_station = initiation_station
        self._process_station = process_station
        self._visualisation_station = visualisation_station

        self._run_finisher = run_finisher

        self._identifier = identifier

        self._tabular_data: Dict[Tuple[DataOrigin, AbstractionLevel], List[pd.DataFrame]] = {}
        self._figure_data: Dict[Tuple[DataOrigin, AbstractionLevel, Column], List[plt.Figure]] = {}

    def process(self):
        """
        Executes the run by processing through the initiation, process, and visualization stations, and then finalizing the run.
        """
        callback_reservoir: RunDataAddProxy = self
        data_reservoir: RunDataGetProxy = self

        self._initiation_station.sign_up_trainer_adapter(self._trainer_adapter)
        self._initiation_station.sign_up_model_adapter(self._model_adapter)
        self._initiation_station.sign_up_module_adapter(self._module_adapter)
        self._initiation_station.process(callback_reservoir)

        self._process_station.sign_up_trainer_adapter(self._trainer_adapter)
        self._process_station.sign_up_model_adapter(self._model_adapter)
        self._process_station.sign_up_module_adapter(self._module_adapter)
        self._process_station.process(callback_reservoir)

        self._visualisation_station.process(callback_reservoir, data_reservoir)

        self._run_finisher.process(data_reservoir)

    def add_df(self, key: Tuple[DataOrigin, AbstractionLevel], data: pd.DataFrame):
        """
        Adds a DataFrame to the run.

        :param key: A tuple containing DataOrigin and AbstractionLevel.
        :param data: The DataFrame to add.
        """
        if key not in self._tabular_data:
            self._tabular_data[key] = []

        self._tabular_data[key].append(data)

    def add_fig(self, key: Tuple[DataOrigin, AbstractionLevel, Column], fig: plt.Figure):
        """
        Adds a figure to the run.

        :param key: A tuple containing DataOrigin, AbstractionLevel, and Column.
        :param fig: The figure to add.
        """
        if key not in self._figure_data:
            self._figure_data[key] = []

        self._figure_data[key].append(fig)

    def get_fig_keys(self) -> List[Tuple[DataOrigin, AbstractionLevel, Column]]:
        """
        Retrieves all figure keys.

        :return: A list of tuples containing DataOrigin, AbstractionLevel, and Column.
        """
        return list(self._figure_data.keys())

    def get_df_keys(self) -> List[Tuple[DataOrigin, AbstractionLevel]]:
        """
        Retrieves all DataFrame keys.

        :return: A list of tuples containing DataOrigin and AbstractionLevel.
        """
        return list(self._tabular_data.keys())

    def get_fig(self, key: Tuple[DataOrigin, AbstractionLevel, Column]) -> List[plt.Figure]:
        """
        Retrieves figures associated with the given key.

        :param key: A tuple containing DataOrigin, AbstractionLevel, and Column.
        :return: A list of figures associated with the key.
        """
        return self._figure_data[key]

    def get_df(self, key: Tuple[DataOrigin, AbstractionLevel]) -> List[pd.DataFrame]:
        """
        Retrieves DataFrames associated with the given key.

        :param key: A tuple containing DataOrigin and AbstractionLevel.
        :return: A list of DataFrames associated with the key.
        """
        return self._tabular_data[key]

    def get_module_adapter(self):
        """
        Retrieves the module adapter.

        :return: The module adapter.
        """
        return self._module_adapter

    def get_model_adapter(self):
        """
        Retrieves the model adapter.

        :return: The model adapter.
        """
        return self._model_adapter

    def get_trainer_adapter(self):
        """
        Retrieves the trainer adapter.

        :return: The trainer adapter.
        """
        return self._trainer_adapter

    def get_process_station(self):
        """
        Retrieves the process station.

        :return: The process station.
        """
        return self._process_station