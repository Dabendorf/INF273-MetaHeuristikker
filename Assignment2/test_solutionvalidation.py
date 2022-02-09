from unittest import TestCase

from Utils import load_problem, feasibility_check, cost_function
import pytest
import random
	
class SolutionValidation(TestCase):
	pytest.paths_testfiles = ["Data/Call_7_Vehicle_3.txt", "Call_18_Vehicle_5.txt", "Call_35_Vehicle_7.txt", "Call_80_Vehicle_20.txt", "Call_130_Vehicle_40.txt", "Call_300_Vehicle_90.txt"]

	def test_costs_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]
		cost = cost_function(sol, pytest.problem_file)

		assert cost == 1871372

	"""def test_costs_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value

	def test_costs_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value

	def test_costs_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value

	def test_costs_file4(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value

	def test_costs_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value"""

	"""def test_feasible_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file4(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True"""

	"""def test_not_feasible_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file4(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False"""
		