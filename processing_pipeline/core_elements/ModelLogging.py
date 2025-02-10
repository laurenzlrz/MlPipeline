from collections import defaultdict
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torchmetrics import Metric

from processing_pipeline.core_elements.AdapterDataKeys import AdapterDataKey
from processing_pipeline.description_enums import Column, ProcessPhase, AbstractionLevel

LOG_FORMAT = "custom_{phase}_{abstraction}_"


class ModelLoggingConnector:
    """
    ModelLoggingConnector is responsible for managing the logging of model metrics during training and evaluation.
    """

    def __init__(self, metrics: dict[Column, type[Metric]]):
        """
        Initializes the ModelLoggingConnector with the given metrics.

        :param metrics: A dictionary mapping Column to Metric types.
        """
        self._metrics: dict[Column, Metric] = {key: metric() for key, metric in metrics.items()}
        self._pl_module: Optional[pl.LightningModule] = None
        self._logs: defaultdict[AdapterDataKey, list] = defaultdict(list)
        self._last_epochs: dict[ProcessPhase, int] = defaultdict(lambda: -1)

    def set_pl_module(self, pl_module: pl.LightningModule):
        """
        Sets the PyTorch Lightning module for logging.

        :param pl_module: An instance of LightningModule.
        """
        self._pl_module = pl_module

    def batch_start(self, target, output, phase: ProcessPhase, batch_idx=None):
        """
        Logs metrics at the start of a batch.

        :param target: The target values.
        :param output: The output values.
        :param phase: The current process phase.
        :param batch_idx: The index of the batch.
        """
        self.calculate_batch(target, output, phase, batch_idx)
        self.calculate_epoch(target, output, phase, batch_idx)
        self.calculate_instance(target, output, phase, batch_idx)

    def check_epoch(self, phase):
        """
        Checks if a new epoch has started and updates the last epoch for the given phase.

        :param phase: The current process phase.
        """
        if self._pl_module.current_epoch > self._last_epochs[phase]:
            self._last_epochs[phase] = self._pl_module.current_epoch
            self.epoch_start(phase)

    def epoch_start(self, phase: ProcessPhase):
        """
        Placeholder method for actions to perform at the start of an epoch.

        :param phase: The current process phase.
        """
        pass

    def calculate_batch(self, target, output, phase, batch_idx):
        """
        Calculates and logs batch metrics.

        :param target: The target values.
        :param output: The output values.
        :param phase: The current process phase.
        :param batch_idx: The index of the batch.
        """
        self.check_epoch(phase)
        target1 = None
        for target in target.values():
            target1 = target
            break

        output1 = None
        for output in output.values():
            output1 = output
            break

        batch_metrics = self.apply_metrics(target1, output1)
        if batch_idx is not None:
            batch_metrics[Column.BATCH_IDX] = batch_idx

        self._log_batch(batch_metrics, phase)

        batch_metrics = self.add_step_and_epoch(batch_metrics)
        self._logs[AdapterDataKey(AbstractionLevel.BATCH, phase)].append(batch_metrics)

    def calculate_epoch(self, target, output, phase, batch_idx):
        """
        Placeholder method for calculating epoch metrics.

        :param target: The target values.
        :param output: The output values.
        :param phase: The current process phase.
        :param batch_idx: The index of the batch.
        """
        pass

    def calculate_instance(self, target, output, phase, batch_idx):
        """
        Placeholder method for calculating instance metrics.

        :param target: The target values.
        :param output: The output values.
        :param phase: The current process phase.
        :param batch_idx: The index of the batch.
        """
        pass

    def apply_metrics(self, target, output):
        """
        Applies the metrics to the target and output values.

        :param target: The target values.
        :param output: The output values.
        :return: A dictionary of calculated metrics.
        """
        results = {}
        for metric_key, metric in self._metrics.items():
            metric: Metric = metric
            metric.update(target, output)
            results[metric_key] = metric.compute()
        return results

    def add_step_and_epoch(self, dict_to_add):
        """
        Adds the current step and epoch to the given dictionary.

        :param dict_to_add: The dictionary to update.
        :return: The updated dictionary.
        """
        step = self._pl_module.global_step
        epoch = self._pl_module.current_epoch
        dict_to_add[Column.GLOBAL_STEP] = step
        dict_to_add[Column.EPOCH] = epoch
        return dict_to_add

    def _log_batch(self, enum_dict: dict, phase):
        """
        Logs batch metrics to the PyTorch Lightning module.

        :param enum_dict: The dictionary of metrics to log.
        :param phase: The current process phase.
        """
        epoch_prefix = LOG_FORMAT.format(phase=phase.value, abstraction=AbstractionLevel.EPOCH.value)
        batch_prefix = LOG_FORMAT.format(phase=phase.value, abstraction=AbstractionLevel.BATCH.value)

        epoch_dict = {epoch_prefix + key.value: value for key, value in enum_dict.items()}
        batch_dict = {batch_prefix + key.value: value for key, value in enum_dict.items()}

        print(f"\n\nstep: {self._pl_module.global_step}, epoch: {self._pl_module.current_epoch}")
        print(f"epoch_dict, model: {epoch_dict}")
        print(f"batch_dict, model {batch_dict}")

        for key, value in epoch_dict.items():
            self._pl_module.log(name=key, value=value, on_epoch=True, on_step=False, prog_bar=False)

        for key, value in batch_dict.items():
            self._pl_module.log(name=key, value=value, on_epoch=False, on_step=True, prog_bar=False)

    def _log_epoch(self, enum_dict: dict, phase):
        """
        Logs epoch metrics to the PyTorch Lightning module.

        :param enum_dict: The dictionary of metrics to log.
        :param phase: The current process phase.
        """
        epoch_prefix = LOG_FORMAT.format(phase=phase.value, abstraction=AbstractionLevel.EPOCH.value)
        epoch_dict = {epoch_prefix + key.value: value for key, value in enum_dict.items()}
        [self._pl_module.log(name=key, value=value, on_epoch=True, on_step=False, prog_bar=False)
         for key, value in epoch_dict.items()]

    def get_last_logs_and_reset(self) -> dict[AdapterDataKey, pd.DataFrame]:
        """
        Retrieves the last logs and resets the internal log storage.

        :return: A dictionary mapping AdapterDataKey to DataFrame containing the last logs.
        """
        df_log_dict = {key: pd.DataFrame(list_of_metrics_dicts) for key, list_of_metrics_dicts in self._logs.items()}
        self._logs.clear()
        return df_log_dict

    def additional_logging(self):
        """
        Placeholder method for additional logging actions.
        """
        pass
