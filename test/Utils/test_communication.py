import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pistacchio.Utils.communication_channel import CommunicationChannel


def test_send_and_receive_from_channel() -> None:
    """This function is used to test the communication channel."""
    channel = CommunicationChannel()
    channel.send_data("Hello World")
    assert channel.receive_data() == "Hello World"
