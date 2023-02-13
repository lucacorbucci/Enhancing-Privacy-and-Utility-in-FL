import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter, sleep

from multiprocess import Manager, Process, Queue

from pistacchio.Components.federated_node import FederatedNode


class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.server_queue = None

    def set_server_queue(self, server_queue):
        self.server_queue = server_queue

    def start_node(self):
        for _ in range(10):
            print(self.server_queue.put(f"Node data from node {self.id}"))


class Server:
    def __init__(self):
        self.server_channel = None

    def set_server_channel(self, server_channel):
        self.server_channel = server_channel

    def start_server(self):
        while True:
            node_data = self.server_channel.get()
            if node_data == "EXIT":
                break
            print(f"Server received: {node_data}")


# mock task that will sleep for a moment
def work(node):
    print("Starting working node")
    node.start_node()
    return f"Task is done: {node.id}"


def start_server(server) -> None:
    server_process = Process(
        target=server.start_server,
    )
    server_process.start()
    return server_process


# entry point
if __name__ == "__main__":
    # create a process pool
    nodes = [Node(i) for i in range(10)]
    server = Server()
    m = multiprocessing.Manager()
    server_channel = m.Queue()

    server.set_server_channel(server_channel)
    server_process = start_server(server)

    for node in nodes:
        node.set_server_queue(server_channel)

    print("Starting nodes")
    with ProcessPoolExecutor(2) as executor:
        # submit tasks and process results
        print("Submitting tasks")
        for result in executor.map(work, nodes):
            print(result)
    server_channel.put("EXIT")

    print("Joining server process")
    server_process.join()
