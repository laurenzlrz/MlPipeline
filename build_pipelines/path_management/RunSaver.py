from typing import List

import pandas as pd

from processing_pipeline.RunDatasetKeys import DFKey, FigKey
from processing_pipeline.PathManager import PathManager
from processing_pipeline.packets.VisualizedRun import VisualizedRun

FIGURE_FORMAT = "fig_{abstraction_level}_{data_origin}_{figure}_{phase}"
DF_FORMAT = "tab_{abstraction_level}_{data_origin}_{phase}"

NAME_SEPARATOR = "_"


def store_df(path_manager: PathManager, data: pd.DataFrame, name: str):
    """
    Stores a DataFrame using the provided PathManager.

    :param path_manager: An instance of PathManager to manage the storage path.
    :param data: The DataFrame to be stored.
    :param name: The name of the DataFrame file.
    """
    if data is not None and len(data) > 0:
        path_manager.save_df(name, data)


def store_fig(path_manager: PathManager, fig, name: str):
    """
    Stores a figure using the provided PathManager.

    :param path_manager: An instance of PathManager to manage the storage path.
    :param fig: The figure to be stored.
    :param name: The name of the figure file.
    """
    if fig is not None:
        path_manager.save_fig(name, fig)


def get_naming(run: VisualizedRun):
    """
    Generates a naming string for the run based on its model adapter name and process type.

    :param run: An instance of VisualizedRun.
    :return: A string representing the name of the run.
    """
    model_adapter_name = run.model_adapter.name
    process_station_name = run.process_type.value
    return f"{model_adapter_name}_{process_station_name}"


class RunSaver:
    """
    RunSaver is responsible for saving visualized runs, including their data frames and figures.
    """

    def __init__(self, root_path: str):
        """
        Initializes the RunSaver with the given root path.

        :param root_path: The root directory for storing runs.
        """
        self.runs = []
        self.path_manager = PathManager(root_path)

    def save(self, run: VisualizedRun):
        """
        Saves the visualized run's data frames and figures to the appropriate paths.

        :param run: An instance of VisualizedRun to be saved.
        """
        path_manager = self.get_run_path_manager(run)

        df_key_list: List[DFKey] = run.get_df_keys()
        for key in df_key_list:
            df = run.get_df(key)
            name = DF_FORMAT.format(abstraction_level=key.abstraction_level.value, data_origin=key.data_origin.value,
                                    phase=key.phase.value)
            store_df(path_manager, df, name)

        fig_key_list: List[FigKey] = run.figure_keys
        for key in fig_key_list:
            fig = run.get_figure(key)
            name = FIGURE_FORMAT.format(abstraction_level=key.abstraction_level.value,
                                        data_origin=key.data_origin.value,
                                        figure=key.column.value, phase=key.phase.value)
            store_fig(path_manager, fig, name)

    def get_run_path_manager(self, run: VisualizedRun) -> PathManager:
        """
        Retrieves the PathManager for the given run, creating necessary directories if they do not exist.

        :param run: An instance of VisualizedRun.
        :return: An instance of PathManager for the run.
        """
        run_dir_name = get_naming(run)
        model_dir = self.path_manager.get_dir_if_exists_or_create(run.model_adapter.name)
        path_manager = PathManager(PathManager(model_dir).create_version_subdirectory(run_dir_name))
        return path_manager
