from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from build_pipelines.path_management.TrainerSaver import TrainerSaver
from processing_pipeline.core_elements.TrainerAdapter import TrainerAdapter
from processing_pipeline.core_elements.TrainerLogging import DataFrameLogger


class TrainerBuilder:
    """
    TrainerBuilder is responsible for constructing TrainerAdapter instances using the provided trainer manager.
    """

    def __init__(self, trainer_manager: TrainerSaver):
        """
        Initializes the TrainerBuilder with the given trainer manager.

        :param trainer_manager: An instance of TrainerSaver to manage trainer paths.
        """
        self._trainer_manager: TrainerSaver = trainer_manager

    def from_config(self, config):
        """
        Constructs a TrainerAdapter from a configuration.

        :param config: Configuration parameters for the trainer.
        """
        pass

    def from_parameters(self, name: str, kwargs) -> TrainerAdapter:
        """
        Constructs a TrainerAdapter using the provided parameters.

        :param name: The name of the trainer.
        :param kwargs: Additional parameters for the trainer.
        :return: An instance of TrainerAdapter.
        """
        tb_logger: TensorBoardLogger = TensorBoardLogger(save_dir=self._trainer_manager.tb_logger_path)
        trainer_logger: DataFrameLogger = DataFrameLogger()
        loggers = [trainer_logger, tb_logger]

        kwargs["logger"] = loggers

        trainer: Trainer = Trainer(**kwargs)

        trainer_adapter: TrainerAdapter = TrainerAdapter(trainer, trainer_logger, name)
        return trainer_adapter

    def from_max_epoch_number(self, name: str, max_epoch_number: int) -> TrainerAdapter:
        """
        Constructs a TrainerAdapter with a specified maximum number of epochs.

        :param name: The name of the trainer.
        :param max_epoch_number: The maximum number of epochs for the trainer.
        :return: An instance of TrainerAdapter.
        """
        return self.from_parameters(name, {"max_epochs": max_epoch_number})
