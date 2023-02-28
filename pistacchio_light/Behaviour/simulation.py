from pistacchio_light.Components.Orchestrator.orchestrator import Orchestrator
from pistacchio_light.Behaviour.environment import manage_environment


class manage_simulation:
    """This class is responsible for performing one full cycle
    of federated training simulation. It trains one model for
    n round, returns the fully trained model and metrics."""

    def __init__(self) -> None:
        return None

    def start_simulation(self) -> None:
        """This is a class that is used for starting the simulation of
        Federated Learning. Once started, it will perform the indicated number
        of rounds, return global model together with local models and
        corresponding metrics."""
        
        configuration = {"num_nodes": 4,
                         "nodes": [1, 2, 3, 4]}
        
        execution_environment = manage_environment(configuration=configuration)
        nodes = execution_environment.initialize_nodes(configuration["nodes"], return_nodes=True)
        print(nodes)
        for node in nodes:
            print(node.node_id)

if __name__ == "__main__":
    manage_simulation().start_simulation()