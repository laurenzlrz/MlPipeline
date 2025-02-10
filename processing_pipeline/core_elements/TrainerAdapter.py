import pandas as pd
from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from processing_pipeline.ICoreElementDoc import IElementDoc
from processing_pipeline.core_elements.TrainerLogging import DataFrameLogger
from processing_pipeline.description_enums import Column


class TrainerAdapter(IElementDoc):
    """
    TrainerAdapter is responsible for adapting a PyTorch Lightning trainer to the processing pipeline,
    managing trainer logging and metadata extraction.
    """

    def __init__(self, trainer: Trainer, trainer_logging: DataFrameLogger, name: str):
        """
        Initializes the TrainerAdapter with the given trainer, logging connector, and name.

        :param trainer: An instance of Trainer representing the PyTorch Lightning trainer.
        :param trainer_logging: An instance of DataFrameLogger for logging trainer data.
        :param name: The name of the trainer adapter.
        """
        super().__init__(name)
        self._trainer: Trainer = trainer
        self._trainer_logging: DataFrameLogger = trainer_logging

        self._hparams = None
        self._last_logs = None

    def get_meta_data(self):
        """
        Retrieves metadata about the trainer, including the maximum number of epochs.

        :return: A DataFrame containing the maximum number of epochs.
        """
        return pd.DataFrame({Column.MAX_EPOCH: [self._trainer.max_epochs]})

    def finalize(self):
        """
        Finalizes the trainer adapter by retrieving the last hyperparameters and logs from the logging connector.
        """
        self._hparams = self._trainer_logging.last_hparams
        self._last_logs = self._trainer_logging.last_logs

    def train(self, model: LightningModule, datamodule: LightningDataModule):
        """
        Trains the model using the given data module.

        :param model: An instance of LightningModule representing the model.
        :param datamodule: An instance of LightningDataModule representing the data module.
        """
        self._trainer.fit(model, datamodule)

    def test(self, model: LightningModule, datamodule: LightningDataModule):
        """
        Tests the model using the given data module.

        :param model: An instance of LightningModule representing the model.
        :param datamodule: An instance of LightningDataModule representing the data module.
        """
        self._trainer.test(model, datamodule)

    @property
    def hparams(self):
        """
        Returns the last hyperparameters retrieved from the logging connector.

        :return: The last hyperparameters.
        """
        return self._hparams

    @property
    def last_logs(self):
        """
        Returns the last logs retrieved from the logging connector.

        :return: The last logs.
        """
        return self._last_logs
