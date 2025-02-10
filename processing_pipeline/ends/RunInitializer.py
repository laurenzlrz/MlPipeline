from processing_pipeline.description_enums import ProcessType
from processing_pipeline.ends.RunFinisher import RunFinisher
from processing_pipeline.packets.OldRun import Run
from processing_pipeline.core_elements.ModuleAdapter import ModuleAdapter
from processing_pipeline.core_elements.ModelAdapter import ModelAdapter
from processing_pipeline.core_elements.TrainerAdapter import TrainerAdapter
from processing_pipeline.packets.StartRun import StartRun
from processing_pipeline.stations.CalculationStation import CalculationStation, StatisticCalculation
from processing_pipeline.stations.InitializingStation import InitializingStation

from processing_pipeline.stations.ProcessStation import TestingStation, TrainingStation
from processing_pipeline.stations.VisualisationStation import VisualisationStation

# Better version of the RunInitializer without maps
PROCESS_MAPPING = {ProcessType.TRAIN: TrainingStation, ProcessType.TEST: TestingStation}


class RunInitializer:
    """
    RunInitializer is responsible for initializing and managing the execution of runs in the processing pipeline.
    """

    def __init__(self, run_finisher: RunFinisher):
        """
        Initializes the RunInitializer with the given RunFinisher.

        :param run_finisher: An instance of RunFinisher used to finalize runs.
        """
        self.run_finisher = run_finisher

        self._model_adapters = {}
        self._module_adapters = {}
        self._trainer_adapters = {}

        self._counter = 0

    def run_from_str(self, model: str, module: str, trainer: str, process: str):
        """
        Initializes and runs a process using string identifiers for the model, module, trainer, and process type.

        :param model: The name of the model adapter.
        :param module: The name of the module adapter.
        :param trainer: The name of the trainer adapter.
        :param process: The process type as a string.
        """
        model_adapter = self._model_adapters[model]
        module_adapter = self._module_adapters[module]
        trainer_adapter = self._trainer_adapters[trainer]
        process_type = ProcessType(process)

        self.run_from_ref(model_adapter, module_adapter, trainer_adapter, process_type)

    def run_from_ref(self, model_adapter: ModelAdapter, module_adapter: ModuleAdapter, trainer_adapter: TrainerAdapter,
                     process: ProcessType):
        """
        Initializes and runs a process using references to the model, module, trainer, and process type.

        :param model_adapter: An instance of ModelAdapter.
        :param module_adapter: An instance of ModuleAdapter.
        :param trainer_adapter: An instance of TrainerAdapter.
        :param process: The process type as an instance of ProcessType.
        """
        process_station = PROCESS_MAPPING[process]()
        initialization_station = InitializingStation()
        visualisation_station = VisualisationStation()
        calculation_station = StatisticCalculation()

        stations = [initialization_station, process_station, visualisation_station, calculation_station,
                    self.run_finisher]

        run = StartRun(model_adapter, module_adapter, trainer_adapter)
        run.update_stations(stations)

        self._counter += 1

        run.next_step()

    def add_module(self, module_adapter: ModuleAdapter):
        """
        Adds a module adapter to the RunInitializer.

        :param module_adapter: An instance of ModuleAdapter to add.
        """
        self._module_adapters[module_adapter.name] = module_adapter

    def add_model(self, model_adapter: ModelAdapter):
        """
        Adds a model adapter to the RunInitializer.

        :param model_adapter: An instance of ModelAdapter to add.
        """
        self._model_adapters[model_adapter.name] = model_adapter

    def add_trainer(self, trainer_adapter: TrainerAdapter):
        """
        Adds a trainer adapter to the RunInitializer.

        :param trainer_adapter: An instance of TrainerAdapter to add.
        """
        self._trainer_adapters[trainer_adapter.name] = trainer_adapter
