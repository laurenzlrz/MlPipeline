from build_pipelines.builder.SchnetModuleBuilder import SchnetModuleBuilder
from build_pipelines.builder.SchnetModelBuilder import SchnetModelBuilder
from build_pipelines.builder.TrainerBuilder import TrainerBuilder


class CoreBuilder:
    """
    CoreBuilder is responsible for constructing models, modules, and trainers using the provided builders.
    """

    def __init__(self, model_builder: SchnetModelBuilder, module_builder: SchnetModuleBuilder,
                 trainer_builder: TrainerBuilder):
        """
        Initializes the CoreBuilder with the given builders.

        :param model_builder: An instance of SchnetModelBuilder.
        :param module_builder: An instance of SchnetModuleBuilder.
        :param trainer_builder: An instance of TrainerBuilder.
        """
        self._model_builder: SchnetModelBuilder = model_builder
        self._module_builder: SchnetModuleBuilder = module_builder
        self._trainer_builder: TrainerBuilder = trainer_builder

        self._model_adapters = {}
        self._module_adapters = {}
        self._trainer_adapters = {}

    def build_model(self, name, db_manager_name, kwargs):
        """
        Builds a model using the model builder and stores it in the model adapters.

        :param name: The name of the model.
        :param db_manager_name: The name of the database manager.
        :param kwargs: Additional parameters for the model.
        :return: The built model adapter.
        """
        db_manager = self._module_builder.get_schnet_manager(db_manager_name)
        model_adapter = self._model_builder.load_with_parameters_and_example_module(name, db_manager, kwargs)
        self._model_adapters[name] = model_adapter
        return model_adapter

    def build_module(self, db_name, module_name, kwargs):
        """
        Builds a module using the module builder and stores it in the module adapters.

        :param db_name: The name of the database.
        :param module_name: The name of the module.
        :param kwargs: Additional parameters for the module.
        :return: The built module adapter.
        """
        module_adapter = self._module_builder.load_from_name(db_name, module_name, kwargs)
        self._module_adapters[module_name] = module_adapter
        return module_adapter

    def build_trainer(self, name, kwargs):
        """
        Builds a trainer using the trainer builder and stores it in the trainer adapters.

        :param name: The name of the trainer.
        :param kwargs: Additional parameters for the trainer.
        :return: The built trainer adapter.
        """
        trainer_adapter = self._trainer_builder.from_parameters(name, kwargs)
        self._trainer_adapters[name] = trainer_adapter
        return trainer_adapter

    def get_model_adapters(self):
        """
        Returns the dictionary of model adapters.

        :return: A dictionary of model adapters.
        """
        return self._model_adapters

    def get_module_adapters(self):
        """
        Returns the dictionary of module adapters.

        :return: A dictionary of module adapters.
        """
        return self._module_adapters

    def get_trainer_adapters(self):
        """
        Returns the dictionary of trainer adapters.

        :return: A dictionary of trainer adapters.
        """
        return self._trainer_adapters
