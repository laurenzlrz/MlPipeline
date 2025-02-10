import pandas as pd
from matplotlib import pyplot as plt

from processing_pipeline.RunDatasetKeys import FigKey
from processing_pipeline.description_enums import AbstractionLevel, Column
from processing_pipeline.packets.ProcessedRun import ProcessedRun
from processing_pipeline.packets.VisualizedRun import VisualizedRun


class VisualisationStation:
    """
    A station in the processing pipeline responsible for visualizing data from a processed run.
    """

    def __init__(self, visualisation_method: () = None):
        """
        Initialize the VisualisationStation with an optional visualization method.

        :param visualisation_method: A callable for custom visualization, taking fig, ax, x_axis, and y_axis as
        parameters.
        """
        self._visualisation_method = visualisation_method

    def process(self, run: ProcessedRun) -> VisualizedRun:
        """
        Process the given ProcessedRun to produce a VisualizedRun with generated figures.

        :param run: The ProcessedRun to process.
        :return: The resulting VisualizedRun with data frames and figures.
        """
        assert isinstance(run, ProcessedRun)

        keys = run.get_df_keys()

        dfs = {key: run.get_df(key) for key in keys}
        epoch_keys = [key for key in keys if key.abstraction_level is AbstractionLevel.EPOCH]

        figures: dict = {}
        for key in epoch_keys:
            epoch_data_df = run.get_df(key)
            new_figures = self.plot_epoch_data(epoch_data_df)
            key_figures: dict[FigKey, plt] = {FigKey(key.data_origin, key.abstraction_level,
                                                     column, key.phase): figure
                                              for column, figure in new_figures.items()}
            figures.update(key_figures)

        return VisualizedRun(run.model_adapter, run.module_adapter, run.trainer_adapter,
                             run.model_metadata, run.module_metadata, run.trainer_metadata,
                             run.process_type, dfs, figures)

    def plot_epoch_data(self, epoch_data: pd.DataFrame):
        """
        Plot the data for each epoch and return the generated figures.

        :param epoch_data: The data frame containing epoch data.
        :return: A dictionary mapping columns to matplotlib figures.
        """
        figures = {}
        for epoch_column in epoch_data.columns:
            if epoch_column is Column.EPOCH:
                continue

            x_axis = epoch_data[Column.EPOCH]
            y_axis = epoch_data[epoch_column]

            fig, ax = plt.subplots()

            if self._visualisation_method is not None:
                self._visualisation_method(fig, ax, x_axis, y_axis)

            # Plot
            ax.plot(x_axis, y_axis, label=epoch_column.value)
            ax.set_xlabel(AbstractionLevel.EPOCH)
            ax.set_ylabel(epoch_column.value)
            ax.set_title(f'{epoch_column.value} vs {AbstractionLevel.EPOCH}')

            ax.legend()
            ax.grid(True)

            figures[epoch_column] = fig
        return figures
