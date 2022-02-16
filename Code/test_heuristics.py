from unittest import TestCase

from Utils import load_problem, feasibility_check, cost_function
import pytest
import random
	
class SolutionValidation(TestCase):
	pytest.paths_testfiles = ["Data/Call_7_Vehicle_3.txt", "Data/Call_18_Vehicle_5.txt", "Data/Call_35_Vehicle_7.txt", "Data/Call_80_Vehicle_20.txt", "Data/Call_130_Vehicle_40.txt", "Data/Call_300_Vehicle_90.txt"]

	"""def test_costs_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]
		cost = cost_function(sol, pytest.problem_file)

		assert cost == 1871372

	def test_feasible_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = [5, 5, 0, 7, 7, 0, 1, 1, 0, 4, 4, 3, 3, 6, 6, 2, 2]
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_not_feasible_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = [1, 1, 0, 7, 7, 0, 5, 5, 0, 6, 6, 2, 2, 4, 4, 3, 3]
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False"""