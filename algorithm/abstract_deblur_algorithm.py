from abc import ABC, abstractmethod

class AbstractDeblurAlgorithm(ABC):

    @abstractmethod
    def run_algorithm(self, **kwargs):
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def should_use(self, algorithm: str) -> bool:
        raise NotImplementedError("Abstract method shall not be invoked!")
