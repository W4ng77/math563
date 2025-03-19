from problem.abstract_deblur_problem import AbstractDeblurProblem

from utils.constants import *


class DeblurProblemL1(AbstractDeblurProblem):
    def run_problem(self, **kwargs):
        pass

    def should_use(self, problem: str) -> bool:
        return problem == L1_PROBLEM
