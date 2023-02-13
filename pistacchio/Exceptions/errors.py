class InvalidDatasetErrorNameError(Exception):
    """This exception is raised when the dataset name is not valid.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Dataset name is not valid") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class InvalidDatasetError(Exception):
    """This exception is raised when the dataset is not valid or
    when it is impossible to load the dataset. This can happen
    when the dataset is not present in the datasets folder or when
    the name of the dataset is not valid.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Dataset is not valid") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class InvalidInitializationError(Exception):
    """This exception is raised when the number of clusters is
    greater than the number of classes in the dataset.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Initialize DataSplit") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class InvalidSplitTypeError(Exception):
    """This exception is raised when the split type is not valid.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Invalid Split Type") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class InvalidSplitConfigurationError(Exception):
    """This exception is raised when the split configuration is not valid.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Invalid splitting configuration") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class MissingConfigurationError(Exception):
    """This exception is raised when you try to use a configuration
    without at least one among server_config and p2p_config.

    Args:
        Exception (_type_)
    """

    def __init__(
        self,
        message: str = "You must specify at least one configuration for the server or the p2p phase",
    ) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class NotYetInitializedPreferencesError(Exception):
    """This exception is raised when you try to use the preferences
    without initializing them.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Preferences not yet initialized") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class NotYetInitializedFederatedLearningError(Exception):
    """This exception is raised when you try to use the federated
    learning without initializing it.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Federated Learning not yet initialized") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class NotYetInitializedServerChannelError(Exception):
    """This exception is raised when you try to use the federated
    learning without initializing it.

    Args:
        Exception (_type_)
    """

    def __init__(self, message: str = "Server Channel not yet initialized") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message}"
