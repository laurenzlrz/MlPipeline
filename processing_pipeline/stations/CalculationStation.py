from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from processing_pipeline.RunDatasetKeys import DFKey, FigKey
from processing_pipeline.description_enums import Column, DataOrigin, AbstractionLevel, ProcessPhase
from processing_pipeline.packets.VisualizedRun import VisualizedRun


class CalculationStation(ABC):
    """
    Abstract base class for calculation stations in the processing pipeline.
    """

    @abstractmethod
    def process(self, run: VisualizedRun) -> VisualizedRun:
        """
        Process the given VisualizedRun.

        :param run: The VisualizedRun to process.
        :return: The processed VisualizedRun.
        """
        pass


class StatisticCalculation(CalculationStation):
    """
    A calculation station that performs statistical calculations on a VisualizedRun.
    """

    def process(self, run: VisualizedRun) -> VisualizedRun:
        """
        Process the given VisualizedRun by performing statistical calculations.

        :param run: The VisualizedRun to process.
        :return: The processed VisualizedRun.
        """
        assert isinstance(run, VisualizedRun)

        for process_phase in ProcessPhase:
            model_logger_key = DFKey(DataOrigin.CALCULATOR, AbstractionLevel.INSTANCE, process_phase)
            if model_logger_key in run.get_df_keys():
                self.show_distribution(run, model_logger_key)

        return run

    def show_distribution(self, run: VisualizedRun, model_logger_key: DFKey):
        """
        Show the distribution of targets and predictions for the given model logger key.

        :param run: The VisualizedRun containing the data.
        :param model_logger_key: The key corresponding to the model logger data frame.
        """
        model_logger_sample_data = run.get_df(model_logger_key)

        targets: pd.Series = model_logger_sample_data[Column.TARGET]
        predictions: pd.Series = model_logger_sample_data[Column.PREDICTED]
        total_error: pd.DataFrame = abs(targets - predictions).to_frame(name=Column.TOTAL_ERROR)
        predicted_distribution = self.scatter_plot(targets, predictions)

        run.add_data_df(DFKey(DataOrigin.CALCULATOR, AbstractionLevel.INSTANCE, ProcessPhase.VALIDATION), total_error)
        run.add_figure(FigKey(DataOrigin.CALCULATOR, AbstractionLevel.INSTANCE,
                              Column.DISTRIBUTION_COMPARISON, model_logger_key.phase), predicted_distribution)

    @staticmethod
    def scatter_plot(target, predicted):
        """
        Create a scatter plot comparing the target and predicted values.

        :param target: The true values.
        :param predicted: The predicted values.
        :return: The matplotlib figure containing the scatter plot.
        """
        plt.figure()
        sns.kdeplot(target, label='True Values', fill=True, alpha=0.5)
        sns.kdeplot(predicted, label='Predicted Values', fill=True, alpha=0.5)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Density Plot of True and Predicted Values')
        plt.legend()
        return plt.gcf()
