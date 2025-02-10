import matplotlib.pyplot as plt
import pandas as pd

from processing_pipeline.RunDatasetKeys import FigKey, DFKey
from processing_pipeline.packets.AbstractPacket import AbstractPacket
from processing_pipeline.packets.InitializedRun import RunMetadataGetProxy
from processing_pipeline.packets.StartRun import RunAdapterGetProxy
from processing_pipeline.packets.ProcessedRun import RunLoggingDataGetProxy


class RunFigureGetProxy:
    """
    A proxy class to handle retrieval of figures in a run.
    """

    def __init__(self, figures: dict[FigKey, plt]):
        """
        Initialize the RunFigureGetProxy with a dictionary of figures.

        :param figures: A dictionary mapping FigKey to matplotlib figures.
        """
        self._figures: dict[FigKey, plt.Figure] = figures

    def get_figure(self, key: FigKey) -> plt.Figure:
        """
        Retrieve a figure by its key.

        :param key: The key corresponding to the figure.
        :return: The matplotlib figure associated with the key.
        """
        return self._figures[key]

    @property
    def figure_keys(self) -> list[FigKey]:
        """
        Get a list of all figure keys.

        :return: A list of all keys in the figures dictionary.
        """
        return list(self._figures.keys())


class VisualizedRun(AbstractPacket, RunAdapterGetProxy, RunMetadataGetProxy, RunLoggingDataGetProxy, RunFigureGetProxy):
    """
    A class representing a visualized run, combining multiple proxies and handling data frames and figures.
    """

    def __init__(self, model_adapter, module_adapter, trainer_adapter,
                 model_metadata, module_metadata, trainer_metadata, process_type, data_dfs, figures: dict[FigKey, plt]):
        """
        Initialize the VisualizedRun with various adapters, metadata, data frames, and figures.

        :param model_adapter: Adapter for the model.
        :param module_adapter: Adapter for the module.
        :param trainer_adapter: Adapter for the trainer.
        :param model_metadata: Metadata for the model.
        :param module_metadata: Metadata for the module.
        :param trainer_metadata: Metadata for the trainer.
        :param process_type: The type of process.
        :param data_dfs: A dictionary mapping DFKey to pandas DataFrames.
        :param figures: A dictionary mapping FigKey to matplotlib figures.
        """
        super().__init__()
        RunAdapterGetProxy.__init__(self, model_adapter, module_adapter, trainer_adapter)
        RunMetadataGetProxy.__init__(self, model_metadata, module_metadata, trainer_metadata)
        RunLoggingDataGetProxy.__init__(self, data_dfs, process_type)
        RunFigureGetProxy.__init__(self, figures)

    def add_data_df(self, key: DFKey, data_df: pd.DataFrame):
        """
        Add a data frame to the run.

        :param key: The key corresponding to the data frame.
        :param data_df: The pandas DataFrame to add.
        """
        self._data_dfs[key] = data_df

    def add_figure(self, key: FigKey, figure: plt.Figure):
        """
        Add a figure to the run.

        :param key: The key corresponding to the figure.
        :param figure: The matplotlib figure to add.
        """
        self._figures[key] = figure
