from pytorch_lightning import LightningDataModule

from processing_pipeline.ICoreElementDoc import IElementDoc


class ModuleAdapter(IElementDoc):
    """
    ModuleAdapter is responsible for adapting a PyTorch Lightning data module to the processing pipeline,
    managing module metadata extraction.
    """

    def __init__(self, module: LightningDataModule, metadata, name: str):
        """
        Initializes the ModuleAdapter with the given data module, metadata, and name.

        :param module: An instance of LightningDataModule representing the data module.
        :param metadata: Metadata associated with the data module.
        :param name: The name of the module adapter.
        """
        super().__init__(name)
        self._module = module
        self._metadata = metadata

    def get_meta_data(self):
        """
        Retrieves metadata about the data module.

        :return: The metadata associated with the data module.
        """
        return self._metadata

    @property
    def module(self) -> LightningDataModule:
        """
        Returns the PyTorch Lightning data module.

        :return: An instance of LightningDataModule.
        """
        return self._module
