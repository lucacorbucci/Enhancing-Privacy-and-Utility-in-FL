from enum import Enum


class TaskType(Enum):
    """Enum class for types of phases of the training."""

    FEDERATEDLEARNING = 1
    CONTRIBUTION = 2
    FAIRNESS = 3


class Task:
    """Definition of a task we want to perform."""

    def __init__(self, task: str) -> None:
        """Initialization of the class Task.
        We need to split the task specified in the json file in two parts
        the first part is the name of the task (federatedlearning, contribution, fairness)
        the second part is an optional additional information that
        can be used while performing the task. For instance if we specify contribution as
        task name we can also specify "loo" or "shapley" as additional information.

        Args:
            task (str): task we want to perform taken from the json file
        """
        splitted_task = task.split("_")
        task_name = splitted_task[0]
        task_info = task.split("_")[1] if len(splitted_task) > 1 else None

        if task_name == "federatedlearning":
            self.task_type = TaskType.FEDERATEDLEARNING
        elif task_name == "contribution":
            self.task_type = TaskType.CONTRIBUTION
        elif task_name == "fairness":
            self.task_type = TaskType.FAIRNESS
        self.task_info = task_info
