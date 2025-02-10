class AbstractPacket:
    """
    AbstractPacket is a base class for packets in the processing pipeline, responsible for managing the sequence of
    stations.
    """

    def __init__(self):
        """
        Initializes the AbstractPacket with an empty list of stations.
        """
        self._stations = []

    def next_step(self):
        """
        Processes the next station in the sequence and updates the packet with the remaining stations.

        If the next station returns a new packet, it updates the stations of the new packet and continues processing.
        """
        # Typing would be also done with generics, in python typing for this phase not possible
        next_station = self._stations.pop(0)
        next_packet: AbstractPacket = next_station.process(self)
        if next_packet is None:
            return
        next_packet.update_stations(self._stations)
        next_packet.next_step()

    def update_stations(self, stations):
        """
        Updates the list of stations for the packet.

        :param stations: A list of stations to be processed.
        """
        self._stations = stations
        # In Java with generics, here would be the station parse process.
