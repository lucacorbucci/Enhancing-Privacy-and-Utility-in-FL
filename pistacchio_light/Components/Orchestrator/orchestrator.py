# Libraries imports
import copy, sys, dill, torch, random, multiprocess, os
# Modules imports
from collections import Counter
from typing import Any
from loguru import logger
from torch import nn
from concurrent.futures import wait
from multiprocess.pool import ThreadPool
# Cross-library imports
from pistacchio_light.Utils.phases import Phase
from pistacchio_light.Components.FederatedNode.federated_node import FederatedNode
from pistacchio_light.Models.federated_model import FederatedModel
from pistacchio_light.Utils.phases import Phase
from pistacchio_light.Utils.preferences import Preferences
from pistacchio_light.Utils.utils import Utils
from pistacchio_light.Models.model_selection import get_model


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<red>{time:YYYY-MM-DD at HH:mm:ss}</red> | {level} | {message}",
)

def connect_node(node, model, communication_queue):
    new_node = copy.deepcopy(node)
    new_node.federated_model = new_node.connect_node(model)
    communication_queue.put(new_node)
    return "OK"


def start_train(node):
    weights = node.train_local_model()
    return (node.node_id, weights)


class Orchestrator:
    def __init__(self, preferences: dict) -> None:
        """Orchestrator is an abstraction object that emulates 
        a classic orchestrator in federated learning. It connects
        to the nodes that are already initialized in the environment
        and performs an indicated number of traning rounds.
        
        Args:
            preferences (dict): Preferences for the training.
        
        Returns:
            None"""
        assert preferences
        self.preferences = preferences
    
    def refresh_environment(self, environemt: dict) -> None:
        """Refreshes the envrionment that is simulated 
        outside the orchestrator. As the environment can change 
        outside the orchestrator (e.g. clients can disconnect or
        malfunction), refresh_envrionment should be called each 
        time we want made any changes to it.
        
        Args:
            environment (dict): Information about the available 
            environment that was previously initialized with the 
            Manage_Environment class.
        
        Returns:
            None"""
        self.environment = environemt

    def launch_orchestrator(self, dataset = None, model = None) -> None:
        """Launches the orchestrator and prepares it for traning.
        This method will invoke self.load_validation_data and
        self.get_model. Upon succesful execution, the initialazed
        orchestrator will load validation data and prepare the model
        indicated in self.preferences["model"]. It is also possible to
        pass a custom model as the function argument.
        
        Args:
            dataset (torch.utils.data.DataLoader[Any]): dataset that we want to use.
            model (nn.Model): model that we want to use."""
        
        # Loads validation data onto the orchestrator instance.
        # If the dataset is not passed as an argument, it will 
        # try to load the dataset assuming standard file directory structure.
        if dataset:
            self.validation_set = dataset
        else:
            self.load_validation_data()
        
        # Creates and load model onto the orchestrator instance.
        # If the model is not passed as an argument, it will try to
        # load the model given the preferences dictionary.
        if model:
            self.orchestrator_model = model
        else:
            self.orchestrator_model = get_model(self.preferences["model"])                

        #if self.preferences.wandb:
            #self.wandb = Utils.configure_wandb(group="Orchestrator", preferences=self.preferences)
        #self.orchestrate_nodes()
        #if self.preferences.wandb:
            #Utils.finish_wandb(wandb_run=self.wandb)
        
    def load_validation_data(self, return_loaded_data = False) -> None:
        """Loads the validation data for the orchestrator.
        By default, it loads data from the disk. Note that the data must
        be stored in a standard file directory structure.
        Standard file structure:
        ./generated_datasets
            /server_validation
                server_validation (binary)
            /test_set
                {node_id}_cluster_0 (binary)
                {node_id}_cluster_0 (binary)
            /train_set
                {node_id}_cluster_0 (binary)
                {node_id}_cluster_0 (binary)
        
        Args:
            return_loaded_data (Bool): If True, method will return the loaded dataset.
            
        Returns:
            None: if return_loaded_data == False
            dataset (torch.utils.data.DataLoader[Any]): if return_loaded_data == True"""
        
        data: torch.utils.data.DataLoader[Any] = None
        with open(
            (
                os.path.join(os.getcwd(), "generated_datasets", self.preferences["dataset"],
                                 "federated_split", "server_validation", "server_validation")
            ), "rb",
            ) as file:
            data = dill.load(file)
            self.validation_set = torch.utils.data.DataLoader(
                data,
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )
        if self.preferences["verbose"] >= 2:
            targets = []
            for _, data in enumerate(self.validation_set, 0):
                targets.append(data[-1])
            targets = [item.item() for sublist in targets for item in sublist]
            logger.info(f"Information from orchestrator: Validation set, loaded: {Counter(targets)}")
        
        if return_loaded_data:
            return self.validation_set
    
    def connect_nodes(self, models_list=None) -> None:
        """Connects already created nods to the orchestrator.
        Note that nodes must be already initialized in the envrionment.
        It is possible to send arbitraty list of models to the nodes using
        model_list. If not provided, orchestrator will copy its model.
        
        Args:
            model_list (list): Optional, a list containing models that we want to
            provide to client. Number of models must equal number of nodes."""

        logger.debug("Connecting available nodes")
        nodes = self.environment['available_clients']

        if models_list:
            raise("Sending models to nodes by providing models_list is not yet implemeneted.")
        # Creating copies of the models to freely modify the weights of each model.
        else:
            model_list = [copy.deepcopy(self.orchestrator_model) for _ in range(len(nodes))]
        
        manager = multiprocess.Manager()
        communication_queue = manager.Queue()

        with multiprocess.Pool(len(nodes)) as pool:
            results = [
                pool.apply_async(connect_node, (node, model, communication_queue))
                for node, model in zip(nodes, model_list)
            ]
            self.connected_nodes = []
            for result in results:
                _ = result.get()
                self.connected_nodes.append(communication_queue.get())

        logger.debug("Nodes connected")
        logger.debug(f"A list of nodes connected: {self.connected_nodes}")

    def simple_protocol(
        self,
    ) -> None:
        logger.debug("Orchestrating nodes...")

        with ThreadPool(self.pool_size) as pool:

            for iteration in range(self.iterations):
                logger.info(f"Iterazione {iteration}")
                weights = {}
                sampled_nodes = random.sample(self.nodes, self.sampled_nodes)
                results = [
                    pool.apply_async(start_train, (node,)) for node in sampled_nodes
                ]

                ready = []

                count = 0
                for result in results:
                    logger.debug(f"Popping {count}")
                    count += 1
                    node_id, model_weights = result.get()

                    weights[node_id] = copy.deepcopy(model_weights.weights)

                avg = Utils.compute_average(weights)
                for node in self.nodes:
                    node.federated_model.update_weights(avg)
                self.federated_model.update_weights(avg)
                
                logger.debug("Computed the average")
                self.log_metrics(iteration=iteration)
        logger.debug("Training finished")

    def log_metrics(self, iteration:int) -> None:
        logger.debug("Computing metrics...")
        (
            loss,
            accuracy,
            fscore,
            precision,
            recall,
            test_accuracy_per_class,
            true_positive_rate,
            false_positive_rate
        ) = self.federated_model.evaluate_model()
        metrics = {"loss":loss, 
                    "accuracy": accuracy, 
                    "fscore": fscore, 
                    "precision": precision,
                    "recall": recall, 
                    "test_accuracy_per_class": test_accuracy_per_class, 
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "epoch": iteration}
        logger.debug(metrics)
        logger.debug("Metrics computed")
        logger.debug("Logging the metrics on wandb")
        if self.preferences.wandb:
            Utils.log_metrics_to_wandb(wandb_run=self.wandb, metrics=metrics)
        logger.debug("Metrics logged")
