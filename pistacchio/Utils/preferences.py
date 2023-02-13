import time

from pistacchio.Exceptions.errors import MissingConfigurationError
from pistacchio.Utils.task import Task


class Preferences:
    """Definition of the class Preferences, used to store the configuration."""

    def __init__(
        self,
        task: str,
        dataset_name: str,
        mode: str,
        debug: bool,
        save_model: bool,
        hyperparameters: dict,
        data_split_config: dict,
        wandb: bool,
        server_config: dict | None = None,
        p2p_config: dict | None = None,
        gpu_config: list | None = None,
        wandb_tags: list | None = ["experiment"],
    ) -> None:
        """Initialization of the class Preferences.

        Args:
            task (str): task we want to perform
            dataset_name (str): name of the dataset we want to use
            mode (str): the architecture we want to use i.e. "semi_p2p", "p2p"
            debug (bool): true if we want to print debug information
            save_model (bool): true if we want to save the model
            hyperparameters (dict): a dictionary containing the hyperparameters
                we want to use during training
            data_split_config (dict): a dictionary containing the configuration for the data split
            wandb (bool): true if we want to use wandb
            server_config (Optional[dict], optional): The configuration we want to use
                for the server. Defaults to None.
            p2p_config (Optional[dict], optional): The configuration we want to use
                during the P2P phase. Defaults to None.
            gpu_config (Optional[list], optional): This paramter is used to configure the GPU that we want to use. For instance if we assign ["cuda_0", "cuda_1"] we will use two GPUs and we will split the clients on them. Defaults to None.
            wandb_tags (Optional[list], optional): a list of tags we want to use on
            wandb. Defaults to ["experiment"].

        Raises
            MissingConfigurationError: This exception is raised when you try to use a configuration
                without at least one among server_config and p2p_config.
        """
        if not server_config and not p2p_config:
            raise MissingConfigurationError

        self.dataset_name = dataset_name
        self.fine_tuning = False
        self.task = Task(task=task)
        self.mode = mode
        self.debug = debug
        self.save_model = save_model
        current_time = time.strftime("%H_%M_%S", time.localtime())
        self.experiment_name = current_time
        self.hyperparameters = hyperparameters
        self.data_split_config = data_split_config
        self.p2p_config = p2p_config
        self.server_config = server_config
        self.wandb = wandb
        self.gpu_config = gpu_config
        self.wandb_tags = wandb_tags
        self.removed_node_id = None

    @staticmethod
    def generate_from_json(data: dict) -> "Preferences":
        """This method is used to generate a Preferences object from a json file.

        Args:
            data (_type_): json file

        Returns
        -------
            Preferences: Preferences object
        """
        return Preferences(
            dataset_name=data["dataset"],
            task=data["task"],
            mode=data["mode"],
            wandb=data["wandb"],
            debug=data["debug"],
            save_model=data["save_model"],
            hyperparameters=data["hyperparameters"],
            data_split_config=data["data_split_config"],
            p2p_config=data.get("p2p_config", None),
            server_config=data.get("server_config", None),
            gpu_config=data.get("gpu_config", None),  # ["cuda:0", "cuda:1"]
            wandb_tags=data.get("wandb_tags", ["experiment"]),
        )
