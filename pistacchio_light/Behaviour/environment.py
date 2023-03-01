from pistacchio_light.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_light.Components.Orchestrator.orchestrator import Orchestrator

class manage_environment:
    """This class is used for managing the Federated Learning environment, e.g.
    starting nodes, shutting down selected nodes, simulating random disconnection etc.
    Can be initialized with preferences dict containing configuraiton."""
    def __init__(self,
                 preferences: dict) -> None:
        """Constructor object of manage_environment class. 
        Args:
        preferences (Preferences): preferences object of the node 
            that contains preference for all nodes.
        Returns:
            None"""
        
        # Creates environment variable that stores information about the environment in which
        # clients reside.
        self.environment = {
            "population": 0, #  Number of all clients that are available in the population.
            "available_clients": list(), #  Number of actually available clients stored as a list.
            "random_dropout_chance": 0, # Chance that a client will dropout from the environment.
            "orchestrator": None # Orchestrator of the whole training
        }
        
        self.preferences = preferences
        self.environment["population"] = preferences["num_nodes"]
    
    def initialize_nodes(self, nodes:list, return_nodes = False) -> dict:
        """Initializes clients in the environment, loads local 
        datset onto clients and returns a list of them."""
        for node_selected in nodes:
            new_node = FederatedNode(
                node_id=node_selected,
                preferences=self.preferences["clients_setting"]["general_clients"],
            )
            if new_node.status == 1:
                self.environment["available_clients"].append(new_node)
            else:
                print(f"Failed to connect client {node_selected}")
        if return_nodes == True:
            return self.environment
    
    def initialize_orchestrator(self):
        self.environment["orchestrator"] = Orchestrator(preferences=self.preferences,\
                                                        environment=self.environment)