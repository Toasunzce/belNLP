from abc import ABC, abstractmethod


class BaseService(ABC):
    """
    Abstract base for all application services.
    Defines the contract between the API layer and business logic.
    Follows Dependency Inversion Principle — routers depend on
    this abstraction, not on concrete service implementations.
    """

    @abstractmethod
    def ready(self) -> bool:
        """Return True if the service is ready to handle requests."""