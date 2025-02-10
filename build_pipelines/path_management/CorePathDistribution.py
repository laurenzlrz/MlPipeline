import os.path

from build_pipelines.path_management.DBSaver import DBSaver
from build_pipelines.path_management.RunSaver import RunSaver
from build_pipelines.path_management.TrainerSaver import TrainerSaver

TRAINER_SAVE_FOLDER = "tb_logger"
METRICS_FOLDER = "metrics"


class CorePathDistribution:
    """
    CorePathDistribution is responsible for managing paths for storing trainers, metrics, and databases.
    """

    def __init__(self, store_root, db_root):
        """
        Initializes the CorePathDistribution with the given store and database roots.

        :param store_root: The root directory for storing trainers and metrics.
        :param db_root: The root directory for storing databases.
        """
        self._store_root = store_root
        self._db_root = db_root

        self._trainer_path = os.path.join(self._store_root, TRAINER_SAVE_FOLDER)
        self._metrics_path = os.path.join(self._store_root, METRICS_FOLDER)
        self._metrics_saver = RunSaver(self._metrics_path)
        self._trainer_saver = TrainerSaver(self._trainer_path)
        self._db_saver = DBSaver(self._db_root)

    @property
    def tb_logger_path(self):
        """
        Returns the path for storing TensorBoard logs.

        :return: The path for TensorBoard logs.
        """
        return self._trainer_path

    @property
    def metrics_saver(self):
        """
        Returns the RunSaver instance for managing metrics.

        :return: An instance of RunSaver for metrics.
        """
        return self._metrics_saver

    @property
    def trainer_saver(self):
        """
        Returns the TrainerSaver instance for managing trainer paths.

        :return: An instance of TrainerSaver.
        """
        return self._trainer_saver

    @property
    def db_saver(self):
        """
        Returns the DBSaver instance for managing database paths.

        :return: An instance of DBSaver.
        """
        return self._db_saver
