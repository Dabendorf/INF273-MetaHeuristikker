from Heuristics import local_search_sim_annealing_latex
from Utils import load_problem, initial_solution, solution_to_ahmed_output, cost_function

import logging

from pdputilslibrary import pdp_load_problem, pdp_feasibility_check, pdp_cost_function

def main():
	logging.basicConfig(filename="run.log", format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	path = "../Data/"
	test_files = ["Call_7_Vehicle_3.txt", "Call_18_Vehicle_5.txt", "Call_35_Vehicle_7.txt", "Call_80_Vehicle_20.txt", "Call_130_Vehicle_40.txt"]
	test_files = ["Call_35_Vehicle_7.txt"]
	# Runs through all test files, performs adaptive algorithm and writes to LaTeX table
	for idx, tf in enumerate(test_files):
		logging.info(f"File: {tf} (file #{idx+1})")
		prob = load_problem(f"{path}{tf}")
		init_sol = initial_solution(problem=prob)

		neighbours = [4, 5, 6, 7, 8, 9]
		num_vehicles, num_calls, best_solution, best_cost, seeds = local_search_sim_annealing_latex(problem=prob, init_sol = init_sol, num_of_iterations=10000, num_of_rounds=10, allowed_neighbours=neighbours, method="aa", file_num=idx+1, statistics=True)
		logging.info(f"Best_solution: {best_solution}")
		print(f"File: {tf} (file #{idx+1})")
		print(f"sol{idx+1} = {best_solution}")
		print(f"{best_cost}")
		logging.info("---------------------------------")

		logging.info("Check with pdp utils")
		pdp_prob = pdp_load_problem(f"{path}{tf}")
		pdp_feasiblity, _ = pdp_feasibility_check(best_solution, pdp_prob)

		pdp_cost = pdp_cost_function(best_solution, pdp_prob)

		if pdp_cost == best_cost and pdp_feasiblity:
			logging.info("pdp utils agrees")
			print("Feasible and correct")
		else:
			logging.error(f"Something went terrible wrong")
			logging.error(f"Feasibility: {pdp_feasiblity}")
			logging.error(f"Cost vs pdp_cost: {best_cost} vs {pdp_cost}")

		logging.info("================================")
		print("================================")
		
if __name__ == "__main__":
	main()