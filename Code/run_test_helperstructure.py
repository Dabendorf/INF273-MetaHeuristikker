from Heuristics import local_search_sim_annealing_latex
from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	#test_files = ["../Data/Call_7_Vehicle_3.txt", "../Data/Call_18_Vehicle_5.txt", "../Data/Call_35_Vehicle_7.txt", "../Data/Call_80_Vehicle_20.txt", "../Data/Call_130_Vehicle_40.txt", "../Data/Call_300_Vehicle_90.txt"]
	test_files = ["../Data/Call_7_Vehicle_3.txt"]

	# Runs through all test files and performs both local search and simulated annealing
	for tf in test_files:
		prob = load_problem(tf)
		init_sol = initial_solution(problem=prob)
		helper = problem_to_helper_structure(problem=prob)

		print(helper)

if __name__ == "__main__":
	main()