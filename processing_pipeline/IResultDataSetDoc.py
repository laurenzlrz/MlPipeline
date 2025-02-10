from typing import Optional

from processing_pipeline.description_enums import DataOrigin, AbstractionLevel, Column, ProcessPhase


class IResultDataSetDoc:
    """
    Class representing a result data set documentation.
    """

    def __init__(self, data_origin: DataOrigin, abstraction_level: AbstractionLevel, phase: ProcessPhase):
        """
        Initialize the IResultDataSetDoc with the specified data origin, abstraction level, and phase.

        :param data_origin: The origin of the data.
        :param abstraction_level: The level of abstraction of the data.
        :param phase: The phase of the process.
        """
        self._data_origin: Optional[DataOrigin] = data_origin
        self._abstraction_level: Optional[AbstractionLevel] = abstraction_level
        self._phase: Optional[ProcessPhase] = phase

    @property
    def data_origin(self) -> DataOrigin:
        """
        Get the origin of the data.

        :return: The data origin.
        """
        return self._data_origin

    @property
    def abstraction_level(self) -> AbstractionLevel:
        """
        Get the level of abstraction of the data.

        :return: The abstraction level.
        """
        return self._abstraction_level

    @property
    def phase(self) -> ProcessPhase:
        """
        Get the phase of the process.

        :return: The process phase.
        """
        return self._phase


class IResultDataElementDoc(IResultDataSetDoc):
    """
    Class representing a result data element documentation, inheriting from IResultDataSetDoc.
    """

    def __init__(self, data_origin: DataOrigin, abstraction_level: AbstractionLevel, column: Column,
                 phase: ProcessPhase):
        """
        Initialize the IResultDataElementDoc with the specified data origin, abstraction level, column, and phase.

        :param data_origin: The origin of the data.
        :param abstraction_level: The level of abstraction of the data.
        :param column: The column of the data.
        :param phase: The phase of the process.
        """
        super().__init__(data_origin, abstraction_level, phase)
        self._column = column

    @property
    def column(self) -> Column:
        """
        Get the column of the data.

        :return: The data column.
        """
        return self._column
