import os
from re import escape, search, match
from typing import Dict, LiteralString

VERSION_FORMAT = "{name}_{i}{type}"
REGEX_TEMPLATE = r"{name}_([\d]+){type}"
FILE_FORMAT = "{name}{type}"

DIR_EXISTS = "Subdirectory {name} already exists"
FILE_NOT_FOUND = "No version found for {name} at path {path}"

CSV_FORMAT = ".csv"
PNG_FORMAT = ".png"
DIR_FORMAT = ""


class PathManager:
    """
    Manages one directory on one level of the file system.
    """

    def __init__(self, root):
        """
        Initialize the PathManager with the specified root directory.

        :param root: The root directory to manage.
        """
        if not os.path.exists(root):
            os.makedirs(root)
        self.root = root

    def get_existing_subdirectory(self, name) -> LiteralString | str | bytes:
        """
        Get the path of an existing subdirectory.

        :param name: The name of the subdirectory.
        :return: The path of the subdirectory.
        :raises FileNotFoundError: If the subdirectory does not exist.
        """
        path = os.path.join(self.root, name)
        if not os.path.exists(path):
            raise FileNotFoundError(FILE_NOT_FOUND.format(name=name, path=self.root))
        return path

    def create_non_existing_subdirectory(self, name) -> LiteralString | str | bytes:
        """
        Create a subdirectory if it does not already exist.

        :param name: The name of the subdirectory.
        :return: The path of the created subdirectory.
        :raises FileExistsError: If the subdirectory already exists.
        """
        path = os.path.join(self.root, name)
        if os.path.exists(path):
            raise FileExistsError(DIR_EXISTS.format(name=name))
        os.makedirs(path)
        return path

    def get_dir_if_exists_or_create(self, name) -> LiteralString | str | bytes:
        """
        Get the path of an existing subdirectory or create it if it does not exist.

        :param name: The name of the subdirectory.
        :return: The path of the subdirectory.
        """
        path = os.path.join(self.root, name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_existing_file(self, name, file_format) -> str:
        """
        Get the path of an existing file.

        :param name: The name of the file.
        :param file_format: The format of the file.
        :return: The path of the file.
        :raises FileNotFoundError: If the file does not exist.
        """
        path = os.path.join(self.root, FILE_FORMAT.format(name=name, type=file_format))
        if not os.path.exists(path):
            raise FileNotFoundError(FILE_NOT_FOUND.format(name=name, path=self.root))
        return path

    def create_version_subdirectory(self, name) -> str:
        """
        Create a versioned subdirectory.

        :param name: The base name of the subdirectory.
        :return: The path of the created subdirectory.
        """

        def type_check(item):
            return os.path.isdir(os.path.join(self.root, item))

        directory = self.get_free_version_entry(name, type_check, DIR_FORMAT)
        os.makedirs(directory)
        return directory

    def create_version_file(self, name, file_format) -> str:
        """
        Create a versioned file.

        :param name: The base name of the file.
        :param file_format: The format of the file.
        :return: The path of the created file.
        """

        def type_check(item):
            return os.path.isfile(os.path.join(self.root, item))

        file_path = self.get_free_version_entry(name, type_check, file_format)
        return file_path

    def get_free_version_entry(self, name, type_check, file_format) -> str:
        """
        Get the path of a free versioned entry.

        :param name: The base name of the entry.
        :param type_check: A function to check the type of the entry.
        :param file_format: The format of the entry.
        :return: The path of the free versioned entry.
        """
        entries = self.list_version_entry(name, type_check, file_format)
        version = 0
        if len(entries.keys()) > 0:
            version = next(i for i in range(1, len(entries.keys()) + 2) if i not in entries.keys())
        entry_name = VERSION_FORMAT.format(name=name, i=version, type=file_format)
        return os.path.join(self.root, entry_name)

    def get_highest_version_subdirectory(self, name) -> str:
        """
        Get the path of the highest versioned subdirectory.

        :param name: The base name of the subdirectory.
        :return: The path of the highest versioned subdirectory.
        :raises FileNotFoundError: If no versioned subdirectory is found.
        """

        def type_check(item):
            return os.path.isdir(os.path.join(self.root, item))

        path = self.get_highest_version_entry(name, type_check, DIR_FORMAT)
        return path

    def get_highest_version_file(self, name, file_format) -> str:
        """
        Get the path of the highest versioned file.

        :param name: The base name of the file.
        :param file_format: The format of the file.
        :return: The path of the highest versioned file.
        :raises FileNotFoundError: If no versioned file is found.
        """

        def type_check(item):
            return os.path.isfile(os.path.join(self.root, item))

        path = self.get_highest_version_entry(name, type_check, file_format)
        return path

    def get_highest_version_entry(self, name, type_check, file_format) -> str:
        """
        Get the path of the highest versioned entry.

        :param name: The base name of the entry.
        :param type_check: A function to check the type of the entry.
        :param file_format: The format of the entry.
        :return: The path of the highest versioned entry.
        :raises FileNotFoundError: If no versioned entry is found.
        """
        sub_items = self.list_version_entry(name, type_check, file_format)
        if len(sub_items.keys()) == 0:
            raise FileNotFoundError(FILE_NOT_FOUND.format(name=name, path=self.root))
        highest = sub_items[max(sub_items.keys())]
        return os.path.join(self.root, highest)

    def list_version_entry(self, name, type_check, file_format) -> Dict[int, str]:
        """
        List all versioned entries.

        :param name: The base name of the entries.
        :param type_check: A function to check the type of the entries.
        :param file_format: The format of the entries.
        :return: A dictionary mapping version numbers to entry names.
        """
        pattern = REGEX_TEMPLATE.format(name=escape(name), type=escape(file_format))
        sub_items = {int(search(pattern, item).group(1)): item
                     for item in os.listdir(self.root)
                     if match(pattern, item) and type_check(item)}
        return sub_items

    def save_version_df(self, name, data):
        """
        Save a DataFrame as a versioned CSV file.

        :param name: The base name of the file.
        :param data: The DataFrame to save.
        """
        path = self.create_version_file(name, CSV_FORMAT)
        data.to_csv(path, index_label="index")

    def save_df(self, name, data):
        """
        Save a DataFrame as a CSV file.

        :param name: The base name of the file.
        :param data: The DataFrame to save.
        :raises FileExistsError: If the file already exists.
        """
        path = os.path.join(self.root, FILE_FORMAT.format(name=name, type=CSV_FORMAT))
        if os.path.exists(path):
            raise FileExistsError(DIR_EXISTS.format(name=name))
        data.to_csv(path, index=False)

    def save_fig(self, name, figure):
        """
        Save a figure as a PNG file.

        :param name: The base name of the file.
        :param figure: The figure to save.
        :raises FileExistsError: If the file already exists.
        """
        path = os.path.join(self.root, FILE_FORMAT.format(name=name, type=PNG_FORMAT))
        if os.path.exists(path):
            raise FileExistsError(DIR_EXISTS.format(name=name))
        figure.savefig(path)

    def save_version_fig(self, name, figure):
        """
        Save a figure as a versioned PNG file.

        :param name: The base name of the file.
        :param figure: The figure to save.
        """
        path = self.create_version_file(name, PNG_FORMAT)
        figure.savefig(path)
