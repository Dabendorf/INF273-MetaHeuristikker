from unittest import TestCase

from Utils import load_problem, feasibility_check, cost_function, initial_solution
from Heuristics import alter_solution_2exchange, local_search, alter_solution_1insert, alter_solution_3exchange
import pytest
import random
	
class HeursticsValidation(TestCase):
	pytest.paths_testfiles = ["Data/Call_7_Vehicle_3.txt", "Data/Call_18_Vehicle_5.txt", "Data/Call_35_Vehicle_7.txt", "Data/Call_80_Vehicle_20.txt", "Data/Call_130_Vehicle_40.txt", "Data/Call_300_Vehicle_90.txt"]

	def test_alter_solution_1insert0a(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		init_sol = initial_solution(problem=pytest.problem_file)
		assert len(init_sol) == len(alter_solution_1insert(pytest.problem_file, init_sol, 0.8))
	
	def test_alter_solution_1insert0b(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		assert len(init_sol) == len(alter_solution_1insert(pytest.problem_file, init_sol, 0.8))

	def test_alter_solution_1insert3a(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		init_sol = initial_solution(problem=pytest.problem_file)
		assert len(init_sol) == len(alter_solution_1insert(pytest.problem_file, init_sol, 0.8))
	
	def test_alter_solution_1insert3b(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		assert len(init_sol) == len(alter_solution_1insert(pytest.problem_file, init_sol, 0.8))

	def test_alter_solution_2exchange1a(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		assert len(init_sol) == len(alter_solution_2exchange(pytest.problem_file, init_sol))

	def test_alter_solution_2exchange1b(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		for i in range(10):
			init_sol = alter_solution_2exchange(pytest.problem_file, init_sol)
		assert len(init_sol) == len(alter_solution_2exchange(pytest.problem_file, init_sol))

	def test_alter_solution_2exchange4a(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		assert len(init_sol) == len(alter_solution_2exchange(pytest.problem_file, init_sol))

	def test_alter_solution_2exchange4b(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		for i in range(10):
			init_sol = alter_solution_2exchange(pytest.problem_file, init_sol)
		assert len(init_sol) == len(alter_solution_2exchange(pytest.problem_file, init_sol))

	def test_alter_solution_3exchange2a(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		for i in range(10):
			init_sol = alter_solution_2exchange(pytest.problem_file, init_sol)
		assert len(init_sol) == len(alter_solution_3exchange(pytest.problem_file, init_sol))

	def test_alter_solution_3exchange2b(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		for i in range(10):
			init_sol = alter_solution_2exchange(pytest.problem_file, init_sol)
		for i in range(10):
			init_sol = alter_solution_3exchange(pytest.problem_file, init_sol)
		assert len(init_sol) == len(alter_solution_3exchange(pytest.problem_file, init_sol))

	def test_alter_solution_3exchange5a(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		for i in range(10):
			init_sol = alter_solution_2exchange(pytest.problem_file, init_sol)
		assert len(init_sol) == len(alter_solution_3exchange(pytest.problem_file, init_sol))

	def test_alter_solution_3exchange5b(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		init_sol = initial_solution(problem=pytest.problem_file)
		for i in range(10):
			init_sol = alter_solution_1insert(pytest.problem_file, init_sol, 0.8)
		for i in range(10):
			init_sol = alter_solution_2exchange(pytest.problem_file, init_sol)
		for i in range(10):
			init_sol = alter_solution_3exchange(pytest.problem_file, init_sol)
		assert len(init_sol) == len(alter_solution_3exchange(pytest.problem_file, init_sol))