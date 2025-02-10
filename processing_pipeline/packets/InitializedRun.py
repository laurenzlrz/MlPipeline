from processing_pipeline.packets.AbstractPacket import AbstractPacket
from processing_pipeline.packets.StartRun import RunAdapterGetProxy


class RunMetadataGetProxy:
    """
    RunMetadataGetProxy is responsible for providing access to metadata for the model, module, and trainer.
    """

    def __init__(self, model_metadata, module_metadata, trainer_metadata):
        """
        Initializes the RunMetadataGetProxy with the given metadata.

        :param model_metadata: Metadata for the model.
        :param module_metadata: Metadata for the module.
        :param trainer_metadata: Metadata for the trainer.
        """
        self._trainer_metadata = trainer_metadata
        self._model_metadata = model_metadata
        self._module_metadata = module_metadata

    @property
    def trainer_metadata(self):
        """
        Returns the trainer metadata.

        :return: The trainer metadata.
        """
        return self._trainer_metadata

    @property
    def model_metadata(self):
        """
        Returns the model metadata.

        :return: The model metadata.
        """
        return self._model_metadata

    @property
    def module_metadata(self):
        """
        Returns the module metadata.

        :return: The module metadata.
        """
        return self._module_metadata


class InitializedRun(AbstractPacket, RunMetadataGetProxy, RunAdapterGetProxy):
    """
    InitializedRun is a packet that combines metadata and adapter proxies for the model, module, and trainer.
    """

    def __init__(self, model_metadata, module_metadata, trainer_metadata, model_adapter, module_adapter,
                 trainer_adapter):
        """
        Initializes the InitializedRun with the given metadata and adapters.

        :param model_metadata: Metadata for the model.
        :param module_metadata: Metadata for the module.
        :param trainer_metadata: Metadata for the trainer.
        :param model_adapter: Adapter for the model.
        :param module_adapter: Adapter for the module.
        :param trainer_adapter: Adapter for the trainer.
        """
        super().__init__()
        RunMetadataGetProxy.__init__(self, model_metadata, module_metadata, trainer_metadata)
        RunAdapterGetProxy.__init__(self, model_adapter, module_adapter, trainer_adapter)
