import torch
from torch import nn

from pistacchio_simulator.Components.Orchestrator.orchestrator import Orchestrator
from pistacchio_simulator.Exceptions.errors import InvalidDatasetErrorNameError
from pistacchio_simulator.Models.celeba import CelebaGenderNet, CelebaNet
from pistacchio_simulator.Models.fashion_mnist import FashionMnistNet
from pistacchio_simulator.Models.mnist import MnistNet
from pistacchio_simulator.Utils.preferences import Preferences
from pistacchio_simulator.Utils.task import Task, TaskType
from multiprocessing import set_start_method

class Experiment:
    # @staticmethod
    # def launch_server(server: Server) -> None:
    #     """This function is used to launch the server.

    #     Args:
    #         server (Server): the server we want to start
    #     """
    #     server.start_server()

    # @staticmethod
    # def launch_federated_node(
    #     node: FederatedNode,
    #     server_queue: CommunicationChannel,
    #     model: nn.Module,
    # ) -> None:
    #     """This function is used to launch a federated node.

    #     Args:
    #         node (FederatedNode): the node we want to start
    #         server_queue (CommunicationChannel): the queue to communicate with the server
    #         model (nn.Module): the model we want to train
    #     """
    #     node.add_server_channel(server_queue)
    #     node.start_node(model)

    # @staticmethod
    # def launch_semi_p2p_experiment(preferences: Preferences, model: nn.Module) -> None:
    #     """This function is used to launch a semi P2P experiment.

    #     Args:
    #         preferences (Preferences): the preferences of the experiment
    #         model (nn.Module): the model we want to train
    #     """
    #     # We create num_clusters clusters
    #     clusters = []
    #     for cluster_id in range(preferences.data_split_config["num_clusters"]):
    #         cluster = P2PCluster(preferences, cluster_id, model)
    #         cluster.init_cluster()
    #         clusters.append(cluster)

    #     # Here we get the channels from all the nodes of each clusters
    #     node_channels = []
    #     for cluster in clusters:
    #         node_channels += cluster.get_channels()
    #     # Launch the server
    #     server = Server(
    #         preferences=preferences,
    #         node_queues=node_channels,
    #         model=copy.deepcopy(model),
    #     )
    #     server_queue = server.get_queue()

    #     # Launch server process
    #     server_process = Process(
    #         target=Experiment.launch_server,
    #         args=(server,),
    #     )
    #     server_process.start()

    #     # Assign the server queue to each node of each cluster
    #     for cluster in clusters:
    #         cluster.start_cluster(server_queue)

    #     for cluster in clusters:
    #         cluster.stop_cluster()

    #     server_process.join()

    # @staticmethod
    # def launch_p2p_experiment(preferences: Preferences, model: nn.Module) -> None:
    #     """This function is used to launch a P2P experiment.

    #     Args:
    #         preferences (Preferences): the preferences of the experiment
    #         model (nn.Module): the model we want to train
    #     """
    #     clusters = []
    #     for cluster_id in range(preferences.data_split_config["num_clusters"]):
    #         cluster = P2PCluster(
    #             preferences,
    #             cluster_id,
    #             model,
    #         )
    #         cluster.init_cluster()
    #         clusters.append(cluster)

    #     # Assign the server queue to each node of each cluster
    #     for cluster in clusters:
    #         cluster.set_neighbors()
    #         cluster.start_cluster_p2p()

    #     for cluster in clusters:
    #         cluster.stop_cluster()

    # @staticmethod
    # def launch_classic_experiment(preferences: Preferences, model: nn.Module) -> None:
    #     """This function is used to launch a classic experiment.

    #     Args:
    #         preferences (Preferences): the preferences of the experiment
    #         model (nn.Module): the model we want to train
    #     """
    #     logging_queue: CommunicationChannel = CommunicationChannel(name="Logging")

    #     nodes = [
    #         FederatedNode(
    #             f"{node_id}_cluster_{cluster_id}",
    #             preferences=preferences,
    #             logging_queue=logging_queue,
    #         )
    #         for node_id in range(preferences.data_split_config["num_nodes"])
    #         for cluster_id in range(preferences.data_split_config["num_clusters"])
    #     ]
    #     nodes_channels = [node.get_communication_channel() for node in nodes]

    #     # Launch the server
    #     server = Server(
    #         preferences=preferences,
    #         node_queues=nodes_channels,
    #         model=copy.deepcopy(model),
    #     )
    #     server_queue = server.get_queue()

    #     # Launch server process
    #     server_process = Process(
    #         target=Experiment.launch_server,
    #         args=(server,),
    #     )
    #     server_process.start()

    #     processes = []
    #     for node in nodes:
    #         process = Process(
    #             target=Experiment.launch_federated_node,
    #             args=(
    #                 node,
    #                 server_queue,
    #                 copy.deepcopy(model),
    #             ),
    #         )
    #         processes.append(process)

    #     for process in processes:
    #         process.start()

    #     server_process.join()
    #     for process in processes:
    #         process.join()

    # @staticmethod
    # def run_experiment(preferences: Preferences, model: nn.Module) -> None:
    #     """This function is used to run an experiment.

    #     Args:
    #         preferences (Preferences): the preferences of the experiment
    #         model (nn.Module): the model we want to train
    #     """
    #     if preferences.mode == "semi_p2p":
    #         Experiment.launch_semi_p2p_experiment(preferences, model)
    #     elif preferences.mode == "p2p":
    #         Experiment.launch_p2p_experiment(preferences, model)
    #     elif preferences.mode == "classic":
    #         Experiment.launch_classic_experiment(preferences, model)
    #     else:
    #         sys.exit()

    # @staticmethod
    # def get_model_to_fine_tune() -> nn.Module:
    #     """This function is used to get the model to fine tune.
    #     In this case we use a pre trained EfficientNet B0 pre trained
    #     on image net.

    #     Returns
    #     -------
    #         nn.Module: the model to fine tune
    #     """
    #     model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    #     for name, param in model.named_parameters(recurse=True):
    #         if not name.startswith("classifier"):
    #             param.requires_grad = False

    #     model.classifier[1] = nn.Linear(in_features=1280, out_features=10)

    #     return model

    @staticmethod
    def get_model(preferences: Preferences) -> nn.Module:
        """This function is used to get the model.

        Returns
        -------
            nn.Module: the model
        """
        model = None
        if preferences.dataset_name == "mnist":
            model = MnistNet()
        elif preferences.dataset_name == "cifar10":
            model = Experiment.get_model_to_fine_tune()
            preferences.fine_tuning = True
        elif preferences.dataset_name == "celeba":
            model = CelebaNet()
        elif preferences.dataset_name == "celeba_gender":
            model = CelebaGenderNet()
        elif preferences.dataset_name == "fashion_mnist":
            model = FashionMnistNet()
        elif preferences.dataset_name == "imaginette":
            model = Experiment.get_model_to_fine_tune()
            preferences.fine_tuning = True
        else:
            raise InvalidDatasetErrorNameError("Invalid dataset name")
        return model

    # @staticmethod
    # def create_server(
    #     preferences: Preferences,
    #     nodes_channels: list[CommunicationChannel],
    #     model: nn.Module,
    # ) -> tuple[Server, CommunicationChannel]:
    #     return Server(
    #         preferences=preferences,
    #         node_queues=nodes_channels,
    #         model=copy.deepcopy(model),
    #     )

    # @staticmethod
    # def create_nodes(
    #     preferences: Preferences,
    # ) -> tuple[list[FederatedNode], list[CommunicationChannel]]:
    #     logging_queue: CommunicationChannel = CommunicationChannel(name="Logging")
    #     return [
    #         FederatedNode(
    #             f"{node_id}_cluster_{cluster_id}",
    #             preferences=preferences,
    #             logging_queue=logging_queue,
    #         )
    #         for node_id in range(preferences.data_split_config["num_nodes"])
    #         for cluster_id in range(preferences.data_split_config["num_clusters"])
    #     ]

    @staticmethod
    def run_federated_learning_experiment(preferences: Preferences) -> None:
        """Run a federated learning experiment.

        Args:
            preferences (Preferences): _description_
        """
        model = Experiment.get_model(preferences)
        # nodes = Experiment.create_nodes(preferences=preferences)
        # server = Experiment.create_server(
        #     preferences=preferences,
        #     model=model,
        #     nodes_channels=[node.get_communication_channel() for node in nodes],
        # )
        orchestrator = Orchestrator(
            preferences=preferences,
            model=model,
        )
        orchestrator.launch_orchestrator()

        # if preferences.hyperparameters.get("noise_multiplier", None):
        #     noise_multipliers = preferences.hyperparameters["noise_multiplier"]
        #     max_grad_norms = preferences.hyperparameters["max_grad_norm"]
        #     p2p_steps = preferences.p2p_config["num_communication_round_pre_training"]
        #     noise_multipliers_p2p = preferences.hyperparameters["noise_multiplier_P2P"]

        #     for noise_multiplier_p2p in noise_multipliers_p2p:
        #         for noise_multiplier in noise_multipliers:
        #             for max_grad_norm in max_grad_norms:
        #                 for p2p_step in p2p_steps:
        #                     preferences.hyperparameters[
        #                         "noise_multiplier"
        #                     ] = noise_multiplier
        #                     preferences.hyperparameters[
        #                         "noise_multiplier_P2P"
        #                     ] = noise_multiplier_p2p
        #                     preferences.hyperparameters["max_grad_norm"] = max_grad_norm
        #                     preferences.p2p_config[
        #                         "num_communication_round_pre_training"
        #                     ] = p2p_step
        #                     Experiment.run_experiment(preferences, model)
        # else:
        #     preferences.p2p_config[
        #         "num_communication_round_pre_training"
        #     ] = preferences.p2p_config["num_communication_round_pre_training"][0]
        #     Experiment.run_experiment(preferences, model)

    # @staticmethod
    # def run_contribution_experiment(preferences: Preferences) -> None:
    #     """Run an experiment to verify the contribution of each
    #     node to the global model.

    #     Args:
    #         preferences (Preferences): preferences specified in the config file

    #     Raises
    #     ------
    #         NotImplementedError: _description_
    #     """
    #     model = Experiment.get_model(preferences)
    #     preferences_ = copy.deepcopy(preferences)
    #     preferences_.task = Task("federatedlearning")
    #     Experiment.launch_classic_experiment(preferences_, model)

    #     logging_queue: CommunicationChannel = CommunicationChannel(name="Logging")

    #     nodes = [
    #         FederatedNode(
    #             f"{node_id}_cluster_{cluster_id}",
    #             preferences=preferences,
    #             logging_queue=logging_queue,
    #         )
    #         for node_id in range(preferences.data_split_config["num_nodes"])
    #         for cluster_id in range(preferences.data_split_config["num_clusters"])
    #     ]
    #     nodes_channels = [node.get_communication_channel() for node in nodes]

    #     for index in range(len(nodes)):
    #         removed_node = nodes.pop(index)
    #         removed_node_id = removed_node.node_id
    #         preferences.removed_node_id = removed_node_id
    #         removed_channel = nodes_channels.pop(index)

    #         # Launch the server
    #         server = Server(
    #             preferences=preferences,
    #             node_queues=nodes_channels,
    #             model=copy.deepcopy(model),
    #         )
    #         server_queue = server.get_queue()

    #         # Launch server process
    #         server_process = Process(
    #             target=Experiment.launch_server,
    #             args=(server,),
    #         )
    #         server_process.start()

    #         processes = []
    #         for node in nodes:
    #             process = Process(
    #                 target=Experiment.launch_federated_node,
    #                 args=(
    #                     node,
    #                     server_queue,
    #                     copy.deepcopy(model),
    #                 ),
    #             )
    #             processes.append(process)

    #         for process in processes:
    #             process.start()

    #         server_process.join()
    #         for process in processes:
    #             process.join()

    #         nodes.insert(index, removed_node)
    #         nodes_channels.insert(index, removed_channel)

    # @staticmethod
    # def run_fairness_experiment(preferences: Preferences) -> None:
    #     """Run an experiment to verify the fairness of the
    #     federated learning model.

    #     Args:
    #         preferences (Preferences): preferences specified in the config file

    #     Raises
    #     ------
    #         NotImplementedError: _description_
    #     """
    #     model = Experiment.get_model(preferences)
    #     Experiment.launch_classic_experiment(preferences, model)

    @staticmethod
    def run(config: dict) -> int:
        """This function is used to prepare all the things
        that are needed to run an experiment.

        Args:
            config (dict): preferences file

        Raises
        ------
            InvalidDatasetErrorNameError: if the dataset name is not valid

        Returns
        -------
            int: 0 if everything went well
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy('file_system')

        preferences = Preferences.generate_from_json(config)
        if preferences.task.task_type == TaskType.FEDERATEDLEARNING:
            Experiment.run_federated_learning_experiment(preferences)
        elif preferences.task.task_type == TaskType.CONTRIBUTION:
            Experiment.run_contribution_experiment(preferences)
        elif preferences.task.task_type == TaskType.FAIRNESS:
            Experiment.run_fairness_experiment(preferences)
        return 0
