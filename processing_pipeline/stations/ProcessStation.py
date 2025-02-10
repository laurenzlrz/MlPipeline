from abc import abstractmethod, ABC
from typing import Dict

import pandas as pd

from processing_pipeline.RunDatasetKeys import DFKey
from processing_pipeline.core_elements.AdapterDataKeys import AdapterDataKey
from processing_pipeline.core_elements.ModelAdapter import ModelAdapter
from processing_pipeline.core_elements.ModuleAdapter import ModuleAdapter
from processing_pipeline.core_elements.TrainerAdapter import TrainerAdapter
from processing_pipeline.description_enums import ProcessType, DataOrigin
from processing_pipeline.packets.InitializedRun import InitializedRun
from processing_pipeline.packets.ProcessedRun import ProcessedRun


def switch_key(data_dict: Dict[AdapterDataKey, pd.DataFrame], origin):
    """
    Switch the keys of the data dictionary to DFKey with the specified origin.

    :param data_dict: A dictionary mapping AdapterDataKey to pandas DataFrames.
    :param origin: The DataOrigin to use for the new keys.
    :return: A dictionary mapping DFKey to pandas DataFrames.
    """
    return {DFKey(origin, key.abstraction_level, key.phase): data for key, data in data_dict.items()}


# TODO Duplicated Code, refactor it in superclass method
class ProcessStation(ABC):
    """
    Abstract base class for process stations in the processing pipeline.
    """

    @abstractmethod
    def process(self, run: InitializedRun) -> ProcessedRun:
        """
        Process the given InitializedRun.

        :param run: The InitializedRun to process.
        :return: The resulting ProcessedRun.
        """
        pass


class TrainingStation(ProcessStation):
    """
    A process station that handles the training phase of the pipeline.
    """

    def __init__(self):
        """
        Initialize the TrainingStation with the process type set to TRAIN.
        """
        self._process_type: ProcessType = ProcessType.TRAIN

    def process(self, run: InitializedRun) -> ProcessedRun:
        """
        Process the given InitializedRun by training the model and module.

        :param run: The InitializedRun to process.
        :return: The resulting ProcessedRun.
        """
        assert isinstance(run, InitializedRun)

        model_adapter: ModelAdapter = run.model_adapter
        module_adapter: ModuleAdapter = run.module_adapter
        trainer_adapter: TrainerAdapter = run.trainer_adapter

        trainer_adapter.train(model_adapter.model, module_adapter.module)

        trainer_adapter.finalize()
        trainer_logging = trainer_adapter.last_logs
        model_adapter.finalize()
        model_logging = model_adapter.last_logs

        if trainer_logging is not None:
            trainer_logging = switch_key(trainer_logging, DataOrigin.TRAINER)
        else:
            trainer_logging = {}

        if model_logging is not None:
            model_logging = switch_key(model_logging, DataOrigin.MODEL)
        else:
            model_logging = {}

        joined_dict = {**trainer_logging, **model_logging}

        return ProcessedRun(model_adapter, module_adapter, trainer_adapter,
                            run.model_metadata, run.module_metadata, run.trainer_metadata,
                            self._process_type, joined_dict)


class TestingStation(ProcessStation):
    """
    A process station that handles the testing phase of the pipeline.
    """

    def __init__(self):
        """
        Initialize the TestingStation with the process type set to TEST.
        """
        self._process_type: ProcessType = ProcessType.TEST

    def process(self, run: InitializedRun) -> ProcessedRun:
        """
        Process the given InitializedRun by testing the model and module.

        :param run: The InitializedRun to process.
        :return: The resulting ProcessedRun.
        """
        assert isinstance(run, InitializedRun)

        model_adapter: ModelAdapter = run.model_adapter
        module_adapter: ModuleAdapter = run.module_adapter
        trainer_adapter: TrainerAdapter = run.trainer_adapter

        trainer_adapter.test(model_adapter.model, module_adapter.module)

        trainer_adapter.finalize()
        trainer_logging = trainer_adapter.last_logs
        model_adapter.finalize()
        model_logging = model_adapter.last_logs

        if trainer_logging is not None:
            trainer_logging = switch_key(trainer_logging, DataOrigin.TRAINER)
        else:
            trainer_logging = {}

        if model_logging is not None:
            model_logging = switch_key(model_logging, DataOrigin.MODEL)
        else:
            model_logging = {}

        joined = {**trainer_logging, **model_logging}

        return ProcessedRun(model_adapter, module_adapter, trainer_adapter,
                            run.model_metadata, run.module_metadata, run.trainer_metadata,
                            self._process_type, joined)
