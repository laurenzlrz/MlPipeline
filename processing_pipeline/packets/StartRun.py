from processing_pipeline.core_elements.ModelAdapter import ModelAdapter
from processing_pipeline.core_elements.ModuleAdapter import ModuleAdapter
from processing_pipeline.core_elements.TrainerAdapter import TrainerAdapter
from processing_pipeline.packets.AbstractPacket import AbstractPacket


class RunAdapterGetProxy:
    """
    RunAdapterGetProxy provides access to the model, module, and trainer adapters.
    """

    def __init__(self, model_adapter: ModelAdapter, module_adapter: ModuleAdapter, trainer_adapter: TrainerAdapter):
        """
        Initializes the RunAdapterGetProxy with the given adapters.

        :param model_adapter: Adapter for the model.
        :param module_adapter: Adapter for the module.
        :param trainer_adapter: Adapter for the trainer.
        """
        self._model_adapter: ModelAdapter = model_adapter
        self._module_adapter: ModuleAdapter = module_adapter
        self._trainer_adapter: TrainerAdapter = trainer_adapter

    @property
    def model_adapter(self) -> ModelAdapter:
        """
        Returns the model adapter.

        :return: The model adapter.
        """
        return self._model_adapter

    @property
    def module_adapter(self) -> ModuleAdapter:
        """
        Returns the module adapter.

        :return: The module adapter.
        """
        return self._module_adapter

    @property
    def trainer_adapter(self) -> TrainerAdapter:
        """
        Returns the trainer adapter.

        :return: The trainer adapter.
        """
        return self._trainer_adapter


class StartRun(AbstractPacket, RunAdapterGetProxy):
    """
    StartRun is a packet that initializes a run with the given model, module, and trainer adapters.
    """

    def __init__(self, model_adapter: ModelAdapter, module_adapter: ModuleAdapter, trainer_adapter: TrainerAdapter):
        """
        Initializes the StartRun with the given adapters.

        :param model_adapter: Adapter for the model.
        :param module_adapter: Adapter for the module.
        :param trainer_adapter: Adapter for the trainer.
        """
        super().__init__()
        RunAdapterGetProxy.__init__(self, model_adapter, module_adapter, trainer_adapter)
