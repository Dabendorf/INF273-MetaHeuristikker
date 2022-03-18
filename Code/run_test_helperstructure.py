from Heuristics import local_search_sim_annealing_latex
from Utils import *
from pprint import pformat

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
		best_sol = [4 , 4 , 7 , 7 , 0 , 2 , 2 , 0 , 1 , 5 , 5 , 3 , 3 , 1 , 0 , 6 , 6]

		success, new_sol = remove_call_from_array(problem=prob, sol=init_sol, call_num=7, vehicle_num=4)
		print(f"New solution: {new_sol}")
		print(f"Removal successful: {success}")
		
		success, new_sol = insert_call_into_array(problem=prob, sol=new_sol, call_num=7, vehicle_num=1)
		print(f"\nAdding one call to vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Insertion successful: {success}")

		success, new_sol = remove_call_from_array(problem=prob, sol=new_sol, call_num=2, vehicle_num=4)
		print(f"\nRemoving one call from vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Removal successful: {success}")
		
		success, new_sol = insert_call_into_array(problem=prob, sol=new_sol, call_num=2, vehicle_num=3)
		print(f"\nAdding one call to vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Insertion successful: {success}")

		success, new_sol = remove_call_from_array(problem=prob, sol=new_sol, call_num=4, vehicle_num=4)
		print(f"\nRemoving one call from vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Removal successful: {success}")
		
		success, new_sol = insert_call_into_array(problem=prob, sol=new_sol, call_num=4, vehicle_num=1)
		print(f"\nAdding one call to vehicle: ")
		print(f"New solution: {new_sol}")
		print(f"Insertion successful: {success}")

if __name__ == "__main__":
	main()