from abc import ABC


class IElementDoc(ABC):
    """
    Abstract base class representing an element documentation.
    """

    def __init__(self, name: str):
        """
        Initialize the IElementDoc with a name.

        :param name: The name of the element.
        """
        self._name = name

    @property
    def name(self) -> str:
        """
        Get the name of the element.

        :return: The name of the element.
        """
        return self._name
