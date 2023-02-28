from pistacchio_light.Components.FederatedNode.federated_node import FederatedNode
class manage_environment:
    """This class is used for managing the Federated Learning environment, e.g.
    starting nodes, shutting down selected nodes, simulating random disconnection etc.
    Can be initialized either with preferences object or dict containing configuraiton."""
    def __init__(self,
                 preferences = None,
                 configuration = None) -> None:
        """Constructor object of manage_environment class. 
        Args:
        preferences (Preferences): preferences object of the node 
            that contains preference for all nodes.
        configuration (dict): configuration object that can be passed
            to the constructor insted of preferences object
            
        Returns:
            None"""
        
        if preferences == None and configuration == None:
            raise("Initializaiton error. You must initialize environment with\
                  preferences object or configuration object.")
        if preferences != None and configuration != None:
            raise("Initialization error, ambigious initialization values.\
                  Please initialize environment with either a preferences \
                  object or configuration object")
        
        #TODO: Initialization from preferences object.
        if preferences:
            pass
        #TODO: Initialization from configuration dictionary.
        else:
            self.num_nodes = configuration["num_nodes"]
    
    def initialize_nodes(self, nodes:list, return_nodes = False) -> list[FederatedNode]:
        """Initializes clients in the environment, loads local 
        datset onto clients and returns a list of them."""
        self.federated_nodes = []
        for node_selected in nodes:
            new_node = FederatedNode(
                node_id=node_selected,
                configuration={"test": 0},
            )
            self.federated_nodes.append(new_node)
        if return_nodes == True:
            return self.federated_nodes