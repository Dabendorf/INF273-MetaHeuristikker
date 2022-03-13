from Heuristics import local_search_sim_annealing_latex
from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	#test_files = ["../Data/Call_7_Vehicle_3.txt", "../Data/Call_18_Vehicle_5.txt", "../Data/Call_35_Vehicle_7.txt", "../Data/Call_80_Vehicle_20.txt", "../Data/Call_130_Vehicle_40.txt", "../Data/Call_300_Vehicle_90.txt"]
	test_files = ["../Data/Call_7_Vehicle_3.txt"]

	# Runs through all test files and performs both local search and simulated annealing
	for tf in test_files:
		prob = load_problem(tf)
		init_sol = initial_solution(problem=prob)
		best_sol = [4 , 4 , 7 , 7 , 0 , 2 , 2 , 0 , 1 , 5 , 5 , 3 , 3 , 1 , 0 , 6 , 6]
		helper_info = problem_to_helper_structure(problem=prob, sol=init_sol)

		print("Initial helper structure: ")
		print(f"New solution: {init_sol}")
		print(f"Helper info: {helper_info}")

		new_sol, helper_info = remove_call_from_array(problem=prob, sol=init_sol, helper_structure=helper_info, call_num=4, vehicle_num=4)
		print(f"\nRemoving one call from vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Helper info: {helper_info}")
		
		new_sol, helper_info = insert_call_into_array(problem=prob, sol=new_sol, helper_structure=helper_info, call_num=4, vehicle_num=1)
		print(f"\nAdding one call to vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Helper info: {helper_info}")

		"""new_sol, helper_info = remove_call_from_array(problem=prob, sol=new_sol, helper_structure=helper_info, call_num=4, vehicle_num=1)
		print(f"\nRemoving one call from vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Helper info: {helper_info}")
		
		new_sol, helper_info = insert_call_into_array(problem=prob, sol=new_sol, helper_structure=helper_info, call_num=4, vehicle_num=2)
		print(f"\nAdding one call to vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Helper info: {helper_info}")"""

if __name__ == "__main__":
	main()