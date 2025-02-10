from argparse import Namespace
from collections import defaultdict
from typing import Union, Any, Optional

import pandas as pd
from pytorch_lightning.loggers import Logger

from processing_pipeline.core_elements.AdapterDataKeys import AdapterDataKey
from processing_pipeline.core_elements.ModelLogging import LOG_FORMAT
from processing_pipeline.description_enums import Column, ProcessPhase, AbstractionLevel

EPOCH_PL_KEY = "epoch"
NAME = "DataFrameLogger"


class DataFrameLogger(Logger):
    """
    DataFrameLogger is responsible for logging metrics and hyperparameters to a DataFrame during training and
    evaluation.
    """

    def __init__(self):
        """
        Initializes the DataFrameLogger.
        """
        super().__init__()
        self._last_hparams = None
        self._last_dfs = None
        self.logs = defaultdict(list)
        self.hparams = []

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """
        Logs the given metrics at the specified step.

        :param metrics: A dictionary of metric names and their values.
        :param step: The current step (optional).
        """
        if EPOCH_PL_KEY not in metrics:
            epoch = -1
        else:
            epoch = metrics[EPOCH_PL_KEY]

        print(f"\n\nepoch trainer: {epoch}")
        print(f"step trainer: {step}")
        print(f"metrics trainer: {metrics}")
        for phase in ProcessPhase:
            for abstraction in AbstractionLevel:
                prefix = LOG_FORMAT.format(phase=phase.value, abstraction=abstraction.value)
                abstraction_metrics = {Column(k.removeprefix(prefix)): v
                                       for k, v in metrics.items() if k.startswith(prefix)}
                if abstraction_metrics:
                    abstraction_metrics[Column.GLOBAL_STEP] = step
                    abstraction_metrics[Column.EPOCH] = epoch
                    self.logs[AdapterDataKey(abstraction, phase)].append(abstraction_metrics)

    def log_hyperparams(self, params: Union[dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        """
        Logs the given hyperparameters.

        :param params: A dictionary or Namespace of hyperparameters.
        """
        self.hparams.append(params)

    def finalize(self, status: str) -> None:
        """
        Finalizes the logging process, storing the last logs and hyperparameters.

        :param status: The status of the training process.
        """
        self._last_dfs = {key: pd.DataFrame(list_of_metrics_dicts) for key, list_of_metrics_dicts in self.logs.items()}
        self._last_hparams = self.hparams
        self.hparams.clear()
        self.logs.clear()

    @property
    def last_logs(self):
        """
        Returns the last logged metrics as DataFrames.

        :return: A dictionary mapping AdapterDataKey to DataFrame containing the last logs.
        """
        return self._last_dfs

    @property
    def last_hparams(self):
        """
        Returns the last logged hyperparameters.

        :return: The last hyperparameters.
        """
        return self._last_hparams

    @property
    def name(self):
        """
        Returns the name of the logger.

        :return: The name of the logger.
        """
        return NAME

    @property
    def version(self):
        """
        Returns the version of the logger.

        :return: The version of the logger.
        """
        return "0.1"
