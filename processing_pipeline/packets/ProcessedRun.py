import pandas as pd

from processing_pipeline.RunDatasetKeys import DFKey
from processing_pipeline.packets.AbstractPacket import AbstractPacket
from processing_pipeline.packets.InitializedRun import RunMetadataGetProxy
from processing_pipeline.packets.StartRun import RunAdapterGetProxy


class RunLoggingDataGetProxy:
    """
    RunLoggingDataGetProxy provides access to logging data and process type for a run.
    """

    def __init__(self, data_dfs: dict[DFKey, pd.DataFrame], process_type):
        """
        Initializes the RunLoggingDataGetProxy with the given data DataFrames and process type.

        :param data_dfs: A dictionary mapping DFKey to pandas DataFrames containing the logging data.
        :param process_type: The type of process being logged.
        """
        self._data_dfs = data_dfs
        self._process_type = process_type

    def get_df(self, key: DFKey) -> pd.DataFrame:
        """
        Returns the DataFrame associated with the given key.

        :param key: The key for the desired DataFrame.
        :return: The DataFrame associated with the given key.
        """
        return self._data_dfs[key]

    def get_df_keys(self) -> list[DFKey]:
        """
        Returns a list of all DataFrame keys.

        :return: A list of DFKey objects.
        """
        return list(self._data_dfs.keys())

    @property
    def process_type(self):
        """
        Returns the process type.

        :return: The process type.
        """
        return self._process_type


class ProcessedRun(AbstractPacket, RunAdapterGetProxy, RunMetadataGetProxy, RunLoggingDataGetProxy):
    """
    ProcessedRun is a packet that combines metadata, adapter proxies, and logging data for a run.
    """

    def __init__(self, model_adapter, module_adapter, trainer_adapter,
                 model_metadata, module_metadata, trainer_metadata, process_type, data_dfs):
        """
        Initializes the ProcessedRun with the given adapters, metadata, process type, and logging data.

        :param model_adapter: Adapter for the model.
        :param module_adapter: Adapter for the module.
        :param trainer_adapter: Adapter for the trainer.
        :param model_metadata: Metadata for the model.
        :param module_metadata: Metadata for the module.
        :param trainer_metadata: Metadata for the trainer.
        :param process_type: The type of process being logged.
        :param data_dfs: A dictionary mapping DFKey to pandas DataFrames containing the logging data.
        """
        super().__init__()
        RunAdapterGetProxy.__init__(self, model_adapter, module_adapter, trainer_adapter)
        RunMetadataGetProxy.__init__(self, model_metadata, module_metadata, trainer_metadata)
        RunLoggingDataGetProxy.__init__(self, data_dfs, process_type)
