import os

from build_pipelines.path_management.DBSaver import DBSaver
from processing_pipeline.core_elements.ModuleAdapter import ModuleAdapter
from schnet_integration.legacy.GeometrySchnetDB import GeometrySchnetDB

DB_FORMAT = ".db"
SPLIT_FORMAT = ".npz"


class SchnetModuleBuilder:
    """
    SchnetModuleBuilder is responsible for constructing module adapters using the provided database saver and module
    parameters.
    """

    def __init__(self, db_saver: DBSaver):
        """
        Initializes the SchnetModuleBuilder with the given database saver.

        :param db_saver: An instance of DBSaver to manage database paths.
        """
        self._db_saver = db_saver
        self._db_managers: dict[str, GeometrySchnetDB] = {}
        self._db_module_adapter: dict[str, ModuleAdapter] = {}
        self._db_modules = {}

    def load_from_name(self, db_name: str, module_name: str, kwargs):
        """
        Loads a module from the given database name and module name.

        :param db_name: The name of the database.
        :param module_name: The name of the module.
        :param kwargs: Additional parameters for the module.
        :return: An instance of ModuleAdapter.
        """
        db_path = self._db_saver.get_path_from_name(db_name, DB_FORMAT)
        return self.load_module(db_name, db_path, module_name, kwargs)

    def load_from_version_name(self, db_name: str, module_name: str, **kwargs):
        """
        Loads a module from the given database version name and module name.

        :param db_name: The name of the database.
        :param module_name: The name of the module.
        :param kwargs: Additional parameters for the module.
        :return: An instance of ModuleAdapter.
        """
        db_path = self._db_saver.get_path_from_version_name(db_name, DB_FORMAT)
        return self.load_module(db_name, db_path, module_name, **kwargs)

    def load_module(self, db_name: str, db_path: str, module_name: str, kwargs):
        """
        Loads a module from the given database path and module name.

        :param db_name: The name of the database.
        :param db_path: The path to the database.
        :param module_name: The name of the module.
        :param kwargs: Additional parameters for the module.
        :return: An instance of ModuleAdapter.
        """
        db_dir_path, db_file_name = os.path.split(db_path)

        if db_path not in self._db_managers:
            self._db_managers[db_path] = GeometrySchnetDB.load_existing(db_file_name, db_dir_path)

        kwargs["split_path"] = self._db_saver.get_split_path(db_name, module_name, SPLIT_FORMAT)
        schnet_module = self._db_managers[db_path].create_schnet_module(**kwargs)
        self._db_modules[module_name] = schnet_module
        module_adapter = ModuleAdapter(schnet_module, {}, module_name)
        self._db_module_adapter[module_name] = module_adapter
        return module_adapter

    def get_schnet_manager(self, db_manager: str):
        """
        Returns the Schnet manager for the given database manager name.

        :param db_manager: The name of the database manager.
        :return: An instance of GeometrySchnetDB.
        """
        db_path = self._db_saver.get_path_from_name(db_manager, DB_FORMAT)
        return self._db_managers[db_path]
