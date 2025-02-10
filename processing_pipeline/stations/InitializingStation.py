from processing_pipeline.packets.InitializedRun import InitializedRun
from processing_pipeline.packets.StartRun import StartRun


class InitializingStation:
    """
    A station in the processing pipeline responsible for initializing a run.
    """

    @staticmethod
    def process(start_run: StartRun) -> InitializedRun:
        """
        Process the given StartRun to produce an InitializedRun.

        :param start_run: The StartRun to process.
        :return: The resulting InitializedRun.
        """
        assert isinstance(start_run, StartRun)

        return InitializedRun(start_run.model_adapter.get_meta_data(), start_run.module_adapter.get_meta_data(),
                              start_run.trainer_adapter.get_meta_data(),
                              start_run.model_adapter, start_run.module_adapter, start_run.trainer_adapter)
