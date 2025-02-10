from build_pipelines.builder.CoreBuilder import CoreBuilder
from build_pipelines.builder.SchnetModuleBuilder import SchnetModuleBuilder
from build_pipelines.builder.SchnetModelBuilder import SchnetModelBuilder
from build_pipelines.path_management.CorePathDistribution import CorePathDistribution
from build_pipelines.builder.TrainerBuilder import TrainerBuilder
from processing_pipeline.ends.RunFinisher import RunFinisher
from processing_pipeline.ends.RunInitializer import RunInitializer


class CoreManager:
    """
    Manages the core components of the pipeline, including builders and initializers.
    """

    def __init__(self, root_store_path: str, db_path: str):
        """
        Initialize the CoreManager with the specified root store path and database path.

        :param root_store_path: The root path for storing data.
        :param db_path: The path to the database.
        """
        self._path_distribution = CorePathDistribution(root_store_path, db_path)
        model_builder = SchnetModelBuilder()
        module_builder = SchnetModuleBuilder(self._path_distribution.db_saver)
        trainer_builder = TrainerBuilder(self._path_distribution.trainer_saver)

        self._core_builder = CoreBuilder(model_builder, module_builder, trainer_builder)
        self._run_finisher = RunFinisher(self._path_distribution.metrics_saver)
        self._run_initializer = RunInitializer(self._run_finisher)

    @property
    def core_builder(self):
        """
        Get the core builder.

        :return: The core builder.
        """
        return self._core_builder

    @property
    def run_initializer(self):
        """
        Get the run initializer.

        :return: The run initializer.
        """
        return self._run_initializer

    def update_adapters(self):
        """
        Update the adapters for models, modules, and trainers in the run initializer.
        """
        for name, adapter in self._core_builder.get_model_adapters().items():
            self._run_initializer.add_model(adapter)

        for name, adapter in self._core_builder.get_module_adapters().items():
            self._run_initializer.add_module(adapter)

        for name, adapter in self._core_builder.get_trainer_adapters().items():
            self._run_initializer.add_trainer(adapter)
