from typing import List, Optional, Any, Type, Dict

import torch
from schnetpack.model import AtomisticModel
from schnetpack.task import AtomisticTask, ModelOutput

from processing_pipeline.core_elements.ModelLogging import ModelLoggingConnector
from processing_pipeline.description_enums import ProcessPhase

SUBSET_PHASE_MAPPING = {"train": ProcessPhase.TRAIN,
                        "val": ProcessPhase.VALIDATION,
                        "test": ProcessPhase.TEST}


class SchnetTaskAdapted(AtomisticTask):

    def __init__(
            self,
            model_logging_connector: ModelLoggingConnector,
            model: AtomisticModel,
            outputs: List[ModelOutput],
            optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_args: Optional[Dict[str, Any]] = None,
            scheduler_cls: Optional[Type] = None,
            scheduler_args: Optional[Dict[str, Any]] = None,
            scheduler_monitor: Optional[str] = None,
            warmup_steps: int = 0,
    ):
        super().__init__(model, outputs, optimizer_cls, optimizer_args,
                         scheduler_cls, scheduler_args, scheduler_monitor, warmup_steps)
        self._model_logging_connector = model_logging_connector

    def log_metrics(self, pred, targets, subset):

        phase = SUBSET_PHASE_MAPPING[subset]
        self._model_logging_connector.batch_start(targets, pred, phase)
