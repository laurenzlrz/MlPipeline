import pytorch_lightning
import torchmetrics

from processing_pipeline.core_elements.ModelAdapter import ModelAdapter
from processing_pipeline.core_elements.ModelLogging import ModelLoggingConnector
from processing_pipeline.description_enums import Column
from schnet_integration.legacy.GeometrySchnetDB import GeometrySchnetDB
from schnet_integration.legacy.SchnetNN import SchnetNN

# Dictionary mapping Column enums to torchmetrics Metric types
METRICS: dict[Column, type[torchmetrics.Metric]] = {
    Column.MAE: torchmetrics.MeanAbsoluteError,
    Column.MSE: torchmetrics.MeanSquaredError,
    Column.NRMSE: torchmetrics.NormalizedRootMeanSquaredError
}


class SchnetModelBuilder:
    """
    SchnetModelBuilder is responsible for constructing model adapters using the provided parameters and example modules.
    """

    def __init__(self):
        """
        Initializes the SchnetModelBuilder with an empty model adapter dictionary.
        """
        self._model_adapter: dict[str, ModelAdapter] = {}

    def load_with_parameters_and_example_module(self, name: str, db_manager: GeometrySchnetDB, kwargs) -> ModelAdapter:
        """
        Loads a model with the given parameters and example module, and returns a ModelAdapter.

        :param name: The name of the model.
        :param db_manager: An instance of GeometrySchnetDB to manage database interactions.
        :param kwargs: Additional parameters for the model.
        :return: An instance of ModelAdapter.
        """
        property_dimensions = db_manager.get_attribute_dimensions()
        additional_input_keys_list = kwargs["additional_input_keys"]
        prediction_keys_list = kwargs["prediction_keys"]

        additional_input_keys_dict = {prop: property_dimensions[prop] for prop in additional_input_keys_list}
        prediction_keys_dict = {prop: property_dimensions[prop] for prop in prediction_keys_list}

        kwargs["additional_input_keys"] = additional_input_keys_dict
        kwargs["prediction_keys"] = prediction_keys_dict

        model: SchnetNN = SchnetNN(**kwargs)
        model_logging_connector = ModelLoggingConnector(METRICS)
        task: pytorch_lightning.LightningModule = model.build_and_return_task(model_logging_connector)
        model_logging_connector.set_pl_module(task)
        model_adapter = ModelAdapter(task, model_logging_connector, name)
        self._model_adapter[name] = model_adapter
        return model_adapter

    def load_with_model(self, name: str, model: pytorch_lightning.LightningModule) -> ModelAdapter:
        """
        Loads a model with the given LightningModule and returns a ModelAdapter.

        :param name: The name of the model.
        :param model: An instance of pytorch_lightning.LightningModule.
        :return: An instance of ModelAdapter.
        """
        model_logging_connector = ModelLoggingConnector(METRICS)
        model_adapter = ModelAdapter(model, model_logging_connector, name)
        self._model_adapter[name] = model_adapter
        return model_adapter
