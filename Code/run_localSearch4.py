from Heuristics import alter_solution_4kinsert, improved_simulated_annealing, local_search_sim_annealing_latex
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
		#init_sol = [4 , 4 , 7 , 7 , 0 , 2 , 2 , 0 , 1 , 5 , 5 , 3 , 3 , 1 , 0 , 6 , 6]

		#print(remove_highest_cost(prob, init_sol))
		#exit(0)

		helper_structure = problem_to_helper_structure(prob, init_sol)

		#alter_solution_4kinsert(prob, init_sol, helper_structure)
		#break
		num_vehicles, num_calls, best_solution, best_cost, seeds = local_search_sim_annealing_latex(problem=prob, init_sol = init_sol, num_of_iterations=10000, num_of_rounds=10, allowed_neighbours=[5, 6, 9], probabilities = [1, 1,1], method="isa")
		overall_best_solution = best_solution
		overall_best_cost = best_cost
		overall_seeds = seeds
		
		"""num_vehicles, num_calls, best_solution, best_cost, seeds = local_search_sim_annealing_latex(problem=prob, init_sol = init_sol, num_of_iterations=10000, num_of_rounds=10, allowed_neighbours=[4,5,6], probabilities = [1/3, 0, 0], method="isa")
		if best_cost < overall_best_cost:
			overall_best_solution = best_solution
			overall_best_cost = best_cost
			overall_seeds = seeds"""

		print(overall_best_cost)
		print(overall_best_solution)
		
		#latex_replace_line(num_vehicles = num_vehicles, num_calls = num_calls, best_solution = overall_best_solution, seeds = overall_seeds)

if __name__ == "__main__":
	main()