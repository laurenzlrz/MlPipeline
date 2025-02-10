from processing_pipeline.description_enums import AbstractionLevel, ProcessPhase


class AdapterDataKey:
    """
    AdapterDataKey is responsible for uniquely identifying data based on abstraction level and process phase.
    """

    def __init__(self, abstraction_level: AbstractionLevel, phase: ProcessPhase):
        """
        Initializes the AdapterDataKey with the given abstraction level and process phase.

        :param abstraction_level: The level of abstraction for the data.
        :param phase: The phase of the process for the data.
        """
        self._abstraction_level = abstraction_level
        self._phase = phase

    def __hash__(self):
        """
        Returns the hash value of the AdapterDataKey.

        :return: The hash value of the AdapterDataKey.
        """
        return hash((self._phase, self._abstraction_level))

    def __eq__(self, other):
        """
        Checks if this AdapterDataKey is equal to another AdapterDataKey.

        :param other: Another instance of AdapterDataKey.
        :return: True if both AdapterDataKey instances are equal, False otherwise.
        """
        if not isinstance(other, AdapterDataKey):
            return False
        return (self._phase, self._abstraction_level) == \
            (self._phase, self._abstraction_level)

    @property
    def abstraction_level(self) -> AbstractionLevel:
        """
        Returns the abstraction level of the AdapterDataKey.

        :return: The abstraction level of the AdapterDataKey.
        """
        return self._abstraction_level

    @property
    def phase(self) -> ProcessPhase:
        """
        Returns the process phase of the AdapterDataKey.

        :return: The process phase of the AdapterDataKey.
        """
        return self._phase
