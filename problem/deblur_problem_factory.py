from problem.abstract_deblur_problem import AbstractDeblurProblem

from problem.deblur_problem_l1 import DeblurProblemL1
from problem.deblur_problem_l2 import DeblurProblemL2

from typing import Optional


class DeblurProblemFactory:

    def __init__(self) -> None:
        self._problems = []
        children = AbstractDeblurProblem.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._problems.append(child())

    """ Public methods """

    def get_problem(self, prob: str) -> Optional[AbstractDeblurProblem]:

        if len(self._problems) == 0:
            return None

        for problem in self._problems:
            if problem.should_use(prob):
                return problem

        return None
