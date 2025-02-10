import schnetpack.transform as trn

from CoreManager import CoreManager
from processing_pipeline.ends.RunInitializer import RunInitializer
from schnet_integration.legacy.MolProperty import MolProperty

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


DB_PATH = "data/dbs"
STORE_PATH = "data/models"

"""
def __init__(self, additional_input_keys: dict[MolProperty, Any], prediction_keys: dict[MolProperty, Any],
                 atom_basis_size=DEF_ATOM_BASIS_SIZE,
                 num_of_interactions=DEF_NUM_OF_INTERACTIONS, rbf_basis_size=DEF_RBF_BASIS_SIZE, cut_off=DEF_CUTOFF,
                 learning_rate=DEF_LEARNING_RATE):
"""

"""
self, selected_properties=None, batch_size=2, num_train=6, num_val=4,
                             transforms=None, num_workers=4, pin_memory=True, split_path=None
"""

"""
    def __init__(
        self,
        *,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[list[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
    ) -> None:
"""

MODEL1 = {"name": "model1", "db_manager_name": "db1", "kwargs": {
    "additional_input_keys": [],
    "prediction_keys": [MolProperty.TOTAL_ENERGY],
}}
MODULE1 = {"module_name": "module1", "db_name": "db1", "kwargs": {
    "selected_properties": [MolProperty.TOTAL_ENERGY],
    "batch_size": 1,
    "num_train": 4,
    "num_val": 2,
    "transforms": [
            trn.ASENeighborList(cutoff=5.),
            trn.CastTo32()
        ]
}}
TRAINER1 = {"name": "trainer1", "kwargs": {
    "max_epochs": 2
}}


def run_1():
    db_path = DB_PATH
    store_path = STORE_PATH

    core_manager = CoreManager(store_path, db_path)
    core_builder = core_manager.core_builder

    core_builder.build_module(**MODULE1)
    core_builder.build_model(**MODEL1)
    core_builder.build_trainer(**TRAINER1)

    core_manager.update_adapters()

    run_initializer: RunInitializer = core_manager.run_initializer
    run_initializer.run_from_str("model1", "module1", "trainer1", "test")


if __name__ == "__main__":
    run_1()
