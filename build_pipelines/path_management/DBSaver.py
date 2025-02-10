from processing_pipeline.PathManager import PathManager

SPLIT_FILE_FORMAT = "{split}_{name}"


class DBSaver:
    """
    DBSaver is responsible for managing database paths and providing methods to retrieve and create paths for
    database files.
    """

    def __init__(self, db_root):
        """
        Initializes the DBSaver with the given database root directory.

        :param db_root: The root directory for storing databases.
        """
        self._db_root = db_root
        self._path_manager = PathManager(self._db_root)

    def get_path_from_name(self, name, db_format):
        """
        Retrieves the path of an existing database file based on its name and format.

        :param name: The name of the database file.
        :param db_format: The format of the database file.
        :return: The path to the existing database file.
        """
        return self._path_manager.get_existing_file(name, db_format)

    def get_path_from_version_name(self, name, db_format):
        """
        Retrieves the path of the highest versioned database file based on its name and format.

        :param name: The name of the database file.
        :param db_format: The format of the database file.
        :return: The path to the highest versioned database file.
        """
        return self._path_manager.get_highest_version_file(name, db_format)

    def get_split_path(self, db_name, module_name, split_file_format):
        """
        Creates and retrieves the path for a split file based on the database name, module name, and split file format.

        :param db_name: The name of the database.
        :param module_name: The name of the module.
        :param split_file_format: The format of the split file.
        :return: The path to the created split file.
        """
        module_dir = self._path_manager.get_dir_if_exists_or_create(db_name)
        module_dir_manager = PathManager(module_dir)

        file_name = SPLIT_FILE_FORMAT.format(split=split_file_format, name=module_name)
        split_path = module_dir_manager.create_version_file(file_name, split_file_format)

        return split_path
