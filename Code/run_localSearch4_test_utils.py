from Heuristics import *
from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	test_files = ["../Data/Call_7_Vehicle_3.txt", "../Data/Call_18_Vehicle_5.txt", "../Data/Call_35_Vehicle_7.txt", "../Data/Call_80_Vehicle_20.txt", "../Data/Call_130_Vehicle_40.txt", "../Data/Call_300_Vehicle_90.txt"]
	test_num = 1
	test_files = test_files[test_num-1:test_num]

	# Runs through all test files and performs both local search and simulated annealing
	for tf in test_files:
		prob = load_problem(tf)
		init_sol = initial_solution(problem=prob)
		sol_a = [[], [6,7,7,6], [1,2,3,2,3,1], [], [4,4,5,5]]
		sol_b = [[4, 4, 7, 7], [2, 2] , [1, 5, 5, 3, 3, 1], [6, 6]]

		"""print(remove_random_call(init_sol, prob, 2))
		print(remove_dummy_call(init_sol, prob, 2))
		print(remove_random_call(sol_a, prob, 2))
		print(remove_dummy_call(sol_a, prob, 2))
		print(remove_random_call(sol_b, prob, 2))
		print(remove_dummy_call(sol_b, prob, 2))"""

		print(remove_highest_cost_call(init_sol, prob, 2))
		print(remove_highest_cost_call(sol_a, prob, 2))
		print(remove_highest_cost_call(sol_b, prob, 2))
if __name__ == "__main__":
	main()