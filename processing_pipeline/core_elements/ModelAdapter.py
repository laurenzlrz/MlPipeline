from typing import Dict

import pandas as pd
from pytorch_lightning import LightningModule

from processing_pipeline.ICoreElementDoc import IElementDoc
from processing_pipeline.core_elements.AdapterDataKeys import AdapterDataKey
from processing_pipeline.core_elements.ModelLogging import ModelLoggingConnector
from processing_pipeline.description_enums import Column


class ModelAdapter(IElementDoc):
    """
    ModelAdapter is responsible for adapting a PyTorch Lightning model to the processing pipeline,
    managing model logging and metadata extraction.
    """

    def __init__(self, model: LightningModule, model_logging_connector: ModelLoggingConnector, name: str):
        """
        Initializes the ModelAdapter with the given model, logging connector, and name.

        :param model: An instance of LightningModule representing the model.
        :param model_logging_connector: An instance of ModelLoggingConnector for logging model data.
        :param name: The name of the model adapter.
        """
        super().__init__(name)
        self._model: LightningModule = model
        self._model_logging_connector: ModelLoggingConnector = model_logging_connector
        self._last_logs = None

    @property
    def model(self) -> LightningModule:
        """
        Returns the PyTorch Lightning model.

        :return: An instance of LightningModule.
        """
        return self._model

    # Provisoric Implementation
    def get_meta_data(self):
        """
        Retrieves metadata about the model, including parameters and metrics.

        :return: A DataFrame containing model parameters and metrics.
        """
        return pd.DataFrame({Column.PARAMS.value: [str(self._model.optimizers)],
                             Column.METRICS.value: [str(self._model)]})

    def finalize(self):
        """
        Finalizes the model adapter by retrieving the last logs from the logging connector and resetting it.
        """
        self._last_logs: dict[AdapterDataKey, pd.DataFrame] = self._model_logging_connector.get_last_logs_and_reset()

    @property
    def last_logs(self) -> Dict[AdapterDataKey, pd.DataFrame]:
        """
        Returns the last logs retrieved from the model logging connector.

        :return: A dictionary mapping AdapterDataKey to DataFrame containing the last logs.
        """
        return self._last_logs
