class TrainerSaver:
    """
    TrainerSaver is responsible for managing paths related to trainer logs.
    """

    def __init__(self, root):
        """
        Initializes the TrainerSaver with the given root directory.

        :param root: The root directory for storing trainer logs.
        """
        self._root = root

    @property
    def tb_logger_path(self):
        """
        Returns the root directory for storing TensorBoard logs.

        :return: The root directory for TensorBoard logs.
        """
        return self._root

    @property
    def tb_logger_name(self):
        """
        Returns the name of the TensorBoard logger directory.

        :return: The name of the TensorBoard logger directory.
        """
        return "tb_logger"
