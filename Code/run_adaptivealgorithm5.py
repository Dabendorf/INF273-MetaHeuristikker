from Heuristics import local_search_sim_annealing_latex
from Utils import load_problem, initial_solution, solution_to_ahmed_output, latex_replace_line

import logging

def main():
	# filename="run.log", 
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	path = "../Data/"
	test_files = ["Call_7_Vehicle_3.txt", "Call_18_Vehicle_5.txt", "Call_35_Vehicle_7.txt", "Call_80_Vehicle_20.txt", "Call_130_Vehicle_40.txt", "Call_300_Vehicle_90.txt"]
	test_num = 1 # TODO remove
	test_files = test_files[test_num-1:test_num]

	# Runs through all test files, performs adaptive algorithm and writes to LaTeX table
	for idx, tf in enumerate(test_files):
		logging.info(f"File: {tf} (file #{idx+1})")
		prob = load_problem(f"{path}{tf}")
		init_sol = initial_solution(problem=prob)

		neighbours = [4, 5, 6, 7, 8, 9]
		num_vehicles, num_calls, best_solution, best_cost, seeds = local_search_sim_annealing_latex(problem=prob, init_sol = init_sol, num_of_iterations=10000, num_of_rounds=10, allowed_neighbours=neighbours, method="aa")
		overall_best_solution = best_solution
		overall_best_cost = best_cost
		overall_seeds = seeds

		logging.info(solution_to_ahmed_output(overall_best_solution))
		
		logging.info(f"Overall best: {overall_best_cost}")
		
		latex_replace_line(num_vehicles = num_vehicles, num_calls = num_calls, best_solution = solution_to_ahmed_output(overall_best_solution), seeds = overall_seeds)

if __name__ == "__main__":
	main()