from abc import ABC, abstractmethod

class AbstractDeblurProblem(ABC):

    @abstractmethod
    def run_problem(self, **kwargs):
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def should_use(self, problem: str) -> bool:
        raise NotImplementedError("Abstract method shall not be invoked!")
