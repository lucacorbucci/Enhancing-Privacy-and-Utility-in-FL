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
        
        # this is a configuration that we can define inside the simulation 
        # class, an alternative to the Preference object.
        preferences = {"num_nodes": 4,
                         "nodes": [1, 2, 3, 4],
                         "dataset": "mnist",
                         "hyperparameters": {
                                "batch_size": 32,
                                "lr": 0.001,
                                "MAX_PHYSICAL_BATCH_SIZE": 128,
                                "DELTA": 1e-5,
                                "noise_multiplier": 0.5,
                                "max_grad_norm": 1.2,
                                "weight_decay": 0,
                                "min_improvement": 0.001,
                                "patience": 10,
                                "min_accuracy": 0.80
                            },
                         "orchestrator_settings":{
                            "local_epochs": 4,
                            "training_rounds": 10,
                            "sampling_size": 4,
                            "differential_privacy_server": False
                            },
                        "clients_setting":{
                            'general_clients': {
                                "test1":0
                            }
                        }
                    }
        
        # This is an execution environment, class that encapsulates all
        # all the 'environmental' variables, included clients and orchestration
        # server. By using manage_environment we can expand our simulation 
        # beyond the nodes and orchestration server abstraction.
        execution_environment = manage_environment(preferences=preferences)
        execution_environment.initialize_nodes(preferences["nodes"])
        execution_environment.initialize_orchestrator()
        print(execution_environment.environment)


if __name__ == "__main__":
    manage_simulation().start_simulation()