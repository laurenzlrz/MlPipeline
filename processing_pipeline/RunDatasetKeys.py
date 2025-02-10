from processing_pipeline.IResultDataSetDoc import IResultDataSetDoc, IResultDataElementDoc


# This file contains the keys used in the processing_pipeline to identify the data, process and origin types of the
# dataframes. TODO: Implement Keys, Change it such that it contains keys and names

class FigKey(IResultDataElementDoc):
    """
    A key class for identifying figures in the processing pipeline.

    Inherits from IResultDataElementDoc.
    """

    def __init__(self, data_origin, abstraction_level, column, phase):
        """
        Initialize the FigKey with the specified parameters.

        :param data_origin: The origin of the data.
        :param abstraction_level: The level of abstraction of the data.
        :param column: The column of the data.
        :param phase: The phase of the process.
        """
        super().__init__(data_origin, abstraction_level, column, phase)

    def __hash__(self):
        """
        Compute the hash value for the FigKey.

        :return: The hash value of the FigKey.
        """
        return hash((self.data_origin, self.abstraction_level, self.column, self.phase))

    def __eq__(self, other):
        """
        Check if this FigKey is equal to another FigKey.

        :param other: The other FigKey to compare with.
        :return: True if the FigKeys are equal, False otherwise.
        """
        if not isinstance(other, FigKey):
            return False
        return (self.data_origin, self.abstraction_level, self.column, self.phase) == \
            (self.data_origin, self.abstraction_level, self.column, self.phase)

    def __repr__(self):
        """
        Get the string representation of the FigKey.

        :return: The string representation of the FigKey.
        """
        return f"Key({self.data_origin}, {self.abstraction_level}, {self.phase})"


class DFKey(IResultDataSetDoc):
    """
    A key class for identifying data frames in the processing pipeline.

    Inherits from IResultDataSetDoc.
    """

    def __init__(self, data_origin, abstraction_level, phase):
        """
        Initialize the DFKey with the specified parameters.

        :param data_origin: The origin of the data.
        :param abstraction_level: The level of abstraction of the data.
        :param phase: The phase of the process.
        """
        super().__init__(data_origin, abstraction_level, phase)

    def __hash__(self):
        """
        Compute the hash value for the DFKey.

        :return: The hash value of the DFKey.
        """
        return hash((self.data_origin, self.abstraction_level, self.phase))

    def __eq__(self, other):
        """
        Check if this DFKey is equal to another DFKey.

        :param other: The other DFKey to compare with.
        :return: True if the DFKeys are equal, False otherwise.
        """
        if not isinstance(other, FigKey):
            return False
        return (self.data_origin, self.abstraction_level, self.phase) == \
            (self.data_origin, self.abstraction_level, self.phase)

    def __repr__(self):
        """
        Get the string representation of the DFKey.

        :return: The string representation of the DFKey.
        """
        return f"Key({self.data_origin}, {self.abstraction_level}, {self.phase})"
