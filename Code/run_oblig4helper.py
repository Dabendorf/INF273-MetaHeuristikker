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
		best_sol = [[4, 4, 7, 7], [2, 2] , [1, 5, 5, 3, 3, 1], [6, 6]]

		print(init_sol)
		print(solution_to_ahmed_output(init_sol))
		print(cost_helper(best_sol[0], prob, 1)+cost_helper(best_sol[1], prob, 2)+cost_helper(best_sol[2], prob, 3)+cost_helper(best_sol[3], prob, 4))
		
		for i in range(4):
			print(feasibility_helper(best_sol[i], prob, i+1))
if __name__ == "__main__":
	main()