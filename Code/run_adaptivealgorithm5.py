from Heuristics import adaptive_algorithm
from Utils import load_problem, initial_solution, solution_to_ahmed_output, latex_replace_line

import logging

def main():
	logging.basicConfig(filename="run.log", format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	test_files = ["../Data/Call_7_Vehicle_3.txt", "../Data/Call_18_Vehicle_5.txt", "../Data/Call_35_Vehicle_7.txt", "../Data/Call_80_Vehicle_20.txt", "../Data/Call_130_Vehicle_40.txt", "../Data/Call_300_Vehicle_90.txt"]
	test_num = 1 # TODO remove
	test_files = test_files[test_num-1:test_num]

	# Runs through all test files, performs adaptive algorithm and writes to LaTeX table
	for tf in test_files:
		logging.info(f"File: {tf}")
		prob = load_problem(tf)
		init_sol = initial_solution(problem=prob)

		neighbours = [4, 5, 6]
		num_vehicles, num_calls, best_solution, best_cost, seeds = adaptive_algorithm(problem=prob, init_sol = init_sol, num_of_iterations=10000, num_of_rounds=10, allowed_neighbours=neighbours, method="aa")
		overall_best_solution = best_solution
		overall_best_cost = best_cost
		overall_seeds = seeds

		logging.info(solution_to_ahmed_output(overall_best_solution))
		
		logging.info(f"Overall best: {overall_best_cost}")
		
		# TODO Add again
		#latex_replace_line(num_vehicles = num_vehicles, num_calls = num_calls, best_solution = overall_best_solution, seeds = overall_seeds)

if __name__ == "__main__":
	main()