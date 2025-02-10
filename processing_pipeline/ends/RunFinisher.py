from build_pipelines.path_management.RunSaver import RunSaver
from processing_pipeline.packets.VisualizedRun import VisualizedRun


class RunFinisher:
    """
    RunFinisher is responsible for finalizing runs by saving them and maintaining a list of processed runs.
    """

    def __init__(self, run_saver: RunSaver):
        """
        Initializes the RunFinisher with the given RunSaver.

        :param run_saver: An instance of RunSaver used to save runs.
        """
        self._run_saver = run_saver
        self.runs = []

    def process(self, run: VisualizedRun):
        """
        Processes the given run by saving it and adding it to the list of processed runs.

        :param run: An instance of VisualizedRun representing the run to be processed.
        """
        self._run_saver.save(run)
        self.runs.append(run)
