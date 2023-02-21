import multiprocessing
from typing import Any


# from multiprocess import Manager, Queue


class CommunicationChannel:
    """Defines a communication channel between two processes."""

    def __init__(self, name: str = "admin") -> None:
        self.name: str = name
        multiprocessing_manager = multiprocessing.Manager()
        self.channel: Queue = multiprocessing_manager.Queue()

    def receive_data(self) -> Any:
        """This function is used to receive data from a queue.

        Returns
        -------
            _type_: the data received from the queue
        """
        return self.channel.get()

    def send_data(self, data: Any) -> None:
        """This function is used to send data in a queue.

        Args:
            data (_type_): the data we want to send
        """
        self.channel.put(data)
