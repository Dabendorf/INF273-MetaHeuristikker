from typing import List
from collections import defaultdict
import logging
from random import randint, randrange, random, choice, seed, choices, sample, shuffle
from timeit import default_timer as timer
import math
import bisect

from nbformat import current_nbformat

from Utils import cost_helper, feasibility_helper, greedy_insert_into_array, merge_vehice_lists, problem_to_helper_structure, insert_call_into_array, remove_call_from_array, remove_highest_cost, split_a_list_at_zeros, cost_function, feasibility_check, latex_add_line, latex_replace_line

logger = logging.getLogger(__name__)

def alter_solution_1insert(problem: dict(), current_solution: List[int], bound_prob_vehicle_vehicle: float) -> List[int]:
	""" 1insert takes a call from one vehicle (including dummy) and puts it into another one"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	logging.debug(f"Alter solution: 1-insert")
	# Two situations: From dummy to vehicle or from vehicle to vehicle
	# Moves from vehicle to vehicle

	found_swap = False
	sol = split_a_list_at_zeros(current_solution)

	while not found_swap:
		if random() > bound_prob_vehicle_vehicle:
			vehicle1 = randint(0,num_vehicles-1)
			vehicle2 = vehicle1
			while vehicle1 == vehicle2:
				vehicle2 = randint(0,num_vehicles-1)

			if len(sol[vehicle2]) > 0:
				found_swap = True
				log_message = f"Move a call from vehicle {vehicle2} to vehicle {vehicle1}"

		# Moves from dummy to vehicle
		else:
			vehicle1 = randint(0,num_vehicles-1)
			vehicle2 = num_vehicles
			try:
				if len(sol[vehicle2]) > 0:
					found_swap = True
					log_message = f"Move a call from dummy vehicle to vehicle {vehicle1}"
			except IndexError:
				bound_prob_vehicle_vehicle = 0

	logging.debug(log_message)

	# Only move calls which are allowed into a vehicle
	# After 10 illegal operations, do it anyway
	call_not_allowed = True
	count_call_iterations = 0
	while call_not_allowed and count_call_iterations < 10:
		call_to_move = choice(sol[vehicle2])
		if call_to_move in vehicle_calls[vehicle1+1]:
			call_not_allowed = False
		else:
			count_call_iterations += 1
			if count_call_iterations == 10:
				logging.debug("Did not swap anything, nothing to get swapped found")
				return current_solution

	sol[vehicle2].remove(call_to_move)
	sol[vehicle2].remove(call_to_move)

	rand_pos1 = randrange(len(sol[vehicle1])+1)
	rand_pos2 = randrange(len(sol[vehicle1])+1)
	sol[vehicle1].insert(rand_pos1, call_to_move)
	sol[vehicle1].insert(rand_pos2, call_to_move)

	new_sol = []
	num_veh_counter = 0
	for el in sol:
		new_sol.extend(el)
		new_sol.append(0)
		num_veh_counter += 1

	if num_veh_counter > num_vehicles:
		new_sol.pop()
	
	return new_sol

def alter_solution_2exchange(problem: dict(), current_solution: List[int]) -> List[int]:
	""" 2exchange swaps one call from one vehicle with another call from another vehicle"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	logging.debug(f"Alter solution: 2-exchange")
	# Select one call from one vehicle and one from another one and exchange them

	found_swap = False
	sol = split_a_list_at_zeros(current_solution)

	counter_swaps = 0
	while not found_swap and counter_swaps < 20:
		vehicle1 = randint(0,num_vehicles-1)
		vehicle2 = vehicle1

		# Make sure there are not the same vehicles
		while vehicle1 == vehicle2:
			vehicle2 = randint(0,num_vehicles-1)

		if len(sol[vehicle2]) > 0 and len(sol[vehicle1]) > 0:
			found_swap = True
			log_message = f"Swap two calls between vehicle {vehicle2} and vehicle {vehicle1}"
		else:
			counter_swaps += 1
			if counter_swaps == 20:
				logging.debug("Did not swap anything, nothing to get swapped found")
				return current_solution
	
	logging.debug(log_message)

	# Only move calls which are allowed into a vehicle
	# After 10 illegal operations, do it anyway
	call_not_allowed = True
	count_call_iterations = 0

	while call_not_allowed and count_call_iterations < 10:
		call_to_move1 = choice(sol[vehicle1])
		call_to_move2 = choice(sol[vehicle2])

		if call_to_move2 in vehicle_calls[vehicle1+1] and call_to_move1 in vehicle_calls[vehicle2+1]:
			call_not_allowed = False
		else:
			count_call_iterations += 1
			if count_call_iterations == 10:
				logging.debug("Did not swap anything, nothing to get swapped found")
				return current_solution

	sol[vehicle2].remove(call_to_move2)
	sol[vehicle2].remove(call_to_move2)
	sol[vehicle1].remove(call_to_move1)
	sol[vehicle1].remove(call_to_move1)

	rand_pos1 = randrange(len(sol[vehicle1])+1)
	rand_pos2 = randrange(len(sol[vehicle1])+1)
	sol[vehicle1].insert(rand_pos1, call_to_move2)
	sol[vehicle1].insert(rand_pos2, call_to_move2)

	rand_pos1 = randrange(len(sol[vehicle2])+1)
	rand_pos2 = randrange(len(sol[vehicle2])+1)
	sol[vehicle2].insert(rand_pos1, call_to_move1)
	sol[vehicle2].insert(rand_pos2, call_to_move1)

	new_sol = []
	num_veh_counter = 0
	for el in sol:
		new_sol.extend(el)
		new_sol.append(0)
		num_veh_counter += 1

	if num_veh_counter > num_vehicles:
		new_sol.pop()
	
	return new_sol

def alter_solution_3exchange(problem: dict(), current_solution: List[int]) -> List[int]:
	""" 3exchange swaps one call each from three different vehicles with each other"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	logging.debug(f"Alter solution: 3-exchange")
	
	found_swap = False
	sol = split_a_list_at_zeros(current_solution)

	counter_swaps = 0
	while not found_swap and counter_swaps < 20:
		vehicle1 = randint(0,num_vehicles-1)
		vehicle2 = vehicle1
		vehicle3 = vehicle2

		# Make sure there are not the same vehicles
		while vehicle1 == vehicle2:
			vehicle2 = randint(0,num_vehicles-1)

		while vehicle3 == vehicle2 or vehicle3 == vehicle1:
			vehicle3 = randint(0,num_vehicles-1)

		if len(sol[vehicle3]) > 0 and len(sol[vehicle2]) > 0 and len(sol[vehicle1]) > 0:
			found_swap = True
			log_message = f"Swap three calls between vehicles {vehicle3}, {vehicle2} and {vehicle1}"
		else:
			counter_swaps += 1
			if counter_swaps == 20:
				logging.debug("Did not swap anything, nothing to get swapped found")
				return current_solution
	
	logging.debug(log_message)

	# Only move calls which are allowed into a vehicle
	# After 10 illegal operations, do it anyway
	call_not_allowed = True
	count_call_iterations = 0

	while call_not_allowed and count_call_iterations < 10:
		call_to_move1 = choice(sol[vehicle1])
		call_to_move2 = choice(sol[vehicle2])
		call_to_move3 = choice(sol[vehicle3])

		if call_to_move3 in vehicle_calls[vehicle2+1] and call_to_move2 in vehicle_calls[vehicle1+1] and call_to_move1 in vehicle_calls[vehicle3+1]:
			call_not_allowed = False
		else:
			count_call_iterations += 1
			if count_call_iterations == 10:
				logging.debug("Did not swap anything, nothing to get swapped found")
				return current_solution

	sol[vehicle3].remove(call_to_move3)
	sol[vehicle3].remove(call_to_move3)
	sol[vehicle2].remove(call_to_move2)
	sol[vehicle2].remove(call_to_move2)
	sol[vehicle1].remove(call_to_move1)
	sol[vehicle1].remove(call_to_move1)

	rand_pos1 = randrange(len(sol[vehicle2])+1)
	rand_pos2 = randrange(len(sol[vehicle2])+1)
	sol[vehicle2].insert(rand_pos1, call_to_move3)
	sol[vehicle2].insert(rand_pos2, call_to_move3)

	rand_pos1 = randrange(len(sol[vehicle1])+1)
	rand_pos2 = randrange(len(sol[vehicle1])+1)
	sol[vehicle1].insert(rand_pos1, call_to_move2)
	sol[vehicle1].insert(rand_pos2, call_to_move2)

	rand_pos1 = randrange(len(sol[vehicle3])+1)
	rand_pos2 = randrange(len(sol[vehicle3])+1)
	sol[vehicle3].insert(rand_pos1, call_to_move1)
	sol[vehicle3].insert(rand_pos2, call_to_move1)

	new_sol = []
	num_veh_counter = 0
	for el in sol:
		new_sol.extend(el)
		new_sol.append(0)
		num_veh_counter += 1

	if num_veh_counter > num_vehicles:
		new_sol.pop()
	
	return new_sol

def alter_solution_4kinsert(problem: dict(), current_solution: List[int], helper_structure) -> List[int]:
	""" """
	#print(current_solution)
	#print("Nr 4")
	#print(f"4: {current_solution}")

	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	# Hyperparameters
	k = 3 # num of inserts
	bound_prob_vehicle_vehicle = 0.8 # probability that dummy doesnt get reinserted
	max_iterations = 3*k # maximum iterations until while breaks
	bound_calls_in_dummy = num_calls//3 # if bigger then take out of dummy
	bound_empty_vehicles = num_vehicles//3
	bound_dummy_prob = 0.3 # bound for dummy if only 1insert
	prob_decrease = 0.4

	iterations = 0
	inserts_done = 0

	logging.debug(f"Alter solution: k-insert")

	dummy_num = num_vehicles

	while iterations < max_iterations and inserts_done < k:
		iterations += 1
		sol = split_a_list_at_zeros(current_solution)
		sol_spread = [len(k)//2 for k in sol]
		
		if sol_spread[-1] > bound_calls_in_dummy or sol_spread.count(0) > bound_empty_vehicles:
			only1insert = True
		else:
			only1insert = False
		try:
			bound = bound_prob_vehicle_vehicle
			if only1insert:
				bound = bound_dummy_prob
			if random() > bound:
				non_empty_lists = [idx for idx, l in enumerate(sol) if len(l) > 0]
			else:
				non_empty_lists = [idx for idx, l in enumerate(sol[:-1]) if len(l) > 0]
			if only1insert:
				veh_to_swap = sample(non_empty_lists, 1)
				empty_lists = [idx for idx, i in enumerate(sol_spread) if i < 3]
				"""print(f"Vehtoswap: {veh_to_swap}")
				print(f"sol {sol}")
				print(f"non empty: {non_empty_lists}")
				print(f"    empty: {empty_lists}")
				print(sol_spread)
				print(f"Sample: {sample(empty_lists, 1)}")"""
				veh_to_swap.extend(sample(empty_lists, 1))
				#print("End method")
			else:
				veh_to_swap = sample(non_empty_lists, 2)
		except ValueError:
			print("Nr 4 Value Error")
			return current_solution
		
		veh_to_swap[0] += 1
		veh_to_swap[1] += 1
		#print(f"Vehicles to swap (index): {veh_to_swap}")
		#print(f"Allowed calls vehicles: {vehicle_calls}")

		if veh_to_swap[0]-1 == dummy_num:
			calls_allowed0 = set(range(1, num_calls+1))
		else:
			calls_allowed0 = vehicle_calls[veh_to_swap[0]]

		if veh_to_swap[1]-1 == dummy_num:
			calls_allowed1 = set(range(1, num_calls+1))
		else:
			calls_allowed1 = vehicle_calls[veh_to_swap[1]]

		"""print(f"Calls allowed for vehicle {veh_to_swap[0]}: {calls_allowed0}")
		print(f"Calls allowed for vehicle {veh_to_swap[1]}: {calls_allowed1}")
		print(f"Current sol for vehicle {veh_to_swap[0]}: {set(sol[veh_to_swap[0]-1])}")
		print(f"Current sol for vehicle {veh_to_swap[1]}: {set(sol[veh_to_swap[1]-1])}")"""

		to_swap_from_0_set = calls_allowed1.intersection(set(sol[veh_to_swap[0]-1]))

		if not only1insert:
			to_swap_from_1_set = calls_allowed0.intersection(set(sol[veh_to_swap[1]-1])) 

		#print(f"To swap from {veh_to_swap[0]}: {to_swap_from_0_set}")
		#if not only1insert:
			#print(f"To swap from {veh_to_swap[1]}: {to_swap_from_1_set}")
		#print(current_solution)

		#print(helper_structure)

		to_swap_from_0 = list(to_swap_from_0_set)

		if not only1insert:
			to_swap_from_1 = list(to_swap_from_1_set)
		

		if not only1insert:
			if len(to_swap_from_0) == 0 or len(to_swap_from_1) == 0:
				#print("ddd")
				continue

		#p = [(call_num, helper_call) for call_num in to_swap_from_0 for helper_call in helper_structure if call_num == helper_call[2]]
		#print(p)
		#q = [call_num for helper_call in helper_structure for call_num in to_swap_from_0 if call_num == helper_call[2]]
		#print(q)
		#print("sss")
		to_swap_from_0 = [call_num for helper_call in helper_structure for call_num in to_swap_from_0 if call_num == helper_call[2]]
		#print(to_swap_from_0)
		if not only1insert:
			to_swap_from_1 = [call_num for helper_call in helper_structure for call_num in to_swap_from_1 if call_num == helper_call[2]]

		#print(f"To swap from {veh_to_swap[0]}: {to_swap_from_0}")
		#print(f"To swap from {veh_to_swap[1]}: {to_swap_from_1}")

		try:
			probabilities = [prob_decrease**i for i in range(0, len(to_swap_from_0))]
			call0 = choices(to_swap_from_0, weights=probabilities)[0]
			#call0 = to_swap_from_0[0]#choice(to_swap_from_0)
			#print(call0)
		except IndexError:
			iterations += 1
			continue
		
		if not only1insert:
			probabilities = [prob_decrease**i for i in range(0, len(to_swap_from_1))]
			call1 = choices(to_swap_from_1, weights=probabilities)[0]
			#call1 = to_swap_from_1[0]#choice(to_swap_from_1)

		#print(f"Call choosen vehicle {veh_to_swap[0]}: {call0}")
		#print(f"Call choosen vehicle {veh_to_swap[1]}: {call1}")

		solution_copy = current_solution.copy()
		
		_, new_sol = remove_call_from_array(problem, solution_copy, call0, veh_to_swap[0])
		successfull, new_sol = insert_call_into_array(problem, new_sol, call0, veh_to_swap[1])

		if not successfull:
			#print("Abbruch")
			break

		if not only1insert:
			_, new_sol = remove_call_from_array(problem, new_sol, call1, veh_to_swap[1])
			successfull, new_sol = insert_call_into_array(problem, new_sol, call1, veh_to_swap[0])
			#print(f"{call0}{call1} {veh_to_swap[0]} {veh_to_swap[1]}")
		"""else:
			print(f"{call0} {veh_to_swap[0]} {veh_to_swap[1]}")
			print("1insert")"""

		#print(datetime.datetime.now().time())
		#print(successfull)
		#print(current_solution)
		if successfull:
			current_solution = new_sol.copy()
			inserts_done += 1
		break
	
	new = current_solution
	if len(current_solution) == new:
		#print("Nr 4 New solution")
		#print(len(new))
		return new
	else:
		#print("Nr 4 Old Solution")
		#print(len(current_solution))
		return current_solution

def alter_solution_regretk(problem: dict(), current_solution: List[int], helper_structure) -> List[int]:
	""" Performs k-regret where one call gets inserted at another position """

	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	k = 2
	sol_split_by_vehicle = split_a_list_at_zeros(current_solution)

	best_costs_for_call = defaultdict(lambda: [])
	call_diff_lookup = dict()
	for call_num in range(1, num_calls+1):
		for veh_num in range(1, num_vehicles+1):
			if call_num in vehicle_calls[veh_num]:
				if call_num not in sol_split_by_vehicle[veh_num-1]:
					veh_cost_original = cost_helper(sol_split_by_vehicle[veh_num-1], problem, veh_num)
					for insert_idx_1 in range(len(sol_split_by_vehicle[veh_num-1])+1):
						temp_call_list = sol_split_by_vehicle[veh_num-1].copy()
						temp_call_list.insert(insert_idx_1, call_num)

						is_feas, _ = feasibility_helper(temp_call_list, problem, veh_num)
						if is_feas:
							for insert_idx_2 in range(1, len(sol_split_by_vehicle[veh_num-1])+2):
								temp_call_list_2 = temp_call_list.copy()
								temp_call_list_2.insert(insert_idx_2, call_num)
								is_feas, _ = feasibility_helper(temp_call_list_2, problem, veh_num)

								if is_feas:
									temp_cost = cost_helper(temp_call_list_2, problem, veh_num)
									temp_diff = temp_cost-veh_cost_original
									bisect.insort(best_costs_for_call[call_num], temp_diff)
									best_costs_for_call[call_num] = best_costs_for_call[call_num][0:k]
									call_diff_lookup[(call_num, temp_diff)] = (veh_num, temp_call_list_2)
	print(best_costs_for_call)
	best_diff = 0
	best_diff_call = -1
	v_0 = -1
	for k, v in best_costs_for_call.items():
		if len(v) > k-1:
			temp_diff = v[k-1] - v[0]
			if temp_diff > best_diff:
				best_diff = temp_diff
				best_diff_call = k
				v_0 = v[0]
	#print(best_diff_call)
	#print(call_diff_lookup)
	#print(call_diff_lookup[(best_diff_call, v_0)])
	if best_diff_call == -1:
		best_diff = float("inf")
		for k, v in best_costs_for_call.items():
			if v[0] < best_diff:
				best_diff = v[0]
				best_diff_call = k
				v_0 = v[0]


	if best_diff_call == -1:
		return current_solution
	
	#_, new_sol = remove_call_from_array(problem, current_solution, call_to_remove, veh_to_remove)
	print(f"Best_diff_call: {best_diff_call}")
	print(f"Old solution: {current_solution}")
	print(f"Current solution before removal: {current_solution}, to_remove: {best_diff_call}")
	current_solution.remove(best_diff_call)
	current_solution.remove(best_diff_call)
	sol_split_by_vehicle = split_a_list_at_zeros(current_solution)
	print(f"Regret: call: {best_diff_call}, vehicle: {call_diff_lookup[(best_diff_call, v_0)]}")

	veh_to_insert, call_list = call_diff_lookup[(best_diff_call, v_0)]
	sol_split_by_vehicle[veh_to_insert-1] = call_list

	new_sol = merge_vehice_lists(sol_split_by_vehicle)
	feas, reason = feasibility_check(new_sol, problem)
	print(feas, reason)
	print(len(new_sol), len(current_solution))
	print(current_solution)
	print(f"NEW SOLUTION: {new_sol}")
	if feas and len(new_sol) == (len(current_solution)+2):
		return new_sol
	else:
		
		print(new_sol)
		print("ERROR")
	"""successfull, new_sol = greedy_insert_into_array(problem, current_solution, best_diff_call, call_diff_lookup[(best_diff_call, v_0)][0])
	print(new_sol)
	if successfull:
		print(new_sol)
		return new_sol
	else:
		print("ERRROR")"""

def alter_solution_greedy_insert_one_vehicle(problem: dict(), current_solution: List[int], helper_structure) -> List[int]:
	""" greedy insertion"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	# Hyperparameters
	bound_prob_vehicle_vehicle = 0.8 # probability that dummy doesnt get reinserted
	max_iterations = 3 # maximum iterations until while breaks

	iterations = 0

	logging.debug(f"Alter solution: greedy insert one vehicle, random removal")

	dummy_num = num_vehicles

	while iterations < max_iterations:
		iterations += 1
		sol = split_a_list_at_zeros(current_solution)
		
		bound = bound_prob_vehicle_vehicle

		# To swap from
		non_empty_lists = [idx for idx, l in enumerate(sol) if len(l) > 0]
		veh_to_swap = sample(non_empty_lists, 1)
		
		if random() > bound:
			end_range = num_vehicles
		else:
			end_range = num_vehicles-1

		veh_to = [veh_to_swap[0]]
		while veh_to[0] == veh_to_swap[0]:
			veh_to = sample(range(0, end_range), 1)
		veh_to_swap.extend(veh_to)
		
		veh_to_swap[0] += 1
		veh_to_swap[1] += 1
		
		# Which calls alre allowed for vehicle 1
		if veh_to_swap[1]-1 == dummy_num:
			calls_allowed1 = set(range(1, num_calls+1))
		else:
			calls_allowed1 = vehicle_calls[veh_to_swap[1]]

		to_swap_from_0_set = calls_allowed1.intersection(set(sol[veh_to_swap[0]-1]))

		"""print(f"Current sol: {sol}")
		print(f"Move one call from {veh_to_swap[0]} to {veh_to_swap[1]}")
		print(f"Calls in 0: {set(sol[veh_to_swap[0]-1])}")
		print(f"Calls allowed 1: {calls_allowed1}")
		print(f"Calls possible to insert: {to_swap_from_0_set}")"""

		if to_swap_from_0_set:
			choosen_call = choice(list(to_swap_from_0_set))
			#print(f"Call choosen: {choosen_call}")

			solution_copy = current_solution.copy()
			_, new_sol = remove_call_from_array(problem, solution_copy, choosen_call, veh_to_swap[0])
			successfull, new_sol = greedy_insert_into_array(problem, new_sol, choosen_call, veh_to_swap[1])

			if successfull:
				current_solution = new_sol.copy()
				break
	
	new = current_solution
	if len(current_solution) == new:
		return new
	else:
		return current_solution

def alter_solution_greedy_insert(problem: dict(), current_solution: List[int], helper_structure) -> List[int]:
	""" greedy insertion"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	# Hyperparameters
	bound_prob_vehicle_vehicle = 0.8 # probability that dummy doesnt get reinserted

	logging.debug(f"Alter solution: k-insert")

	dummy_num = num_vehicles

	original_sol = current_solution.copy()

	best_cost = cost_function(current_solution, problem)

	sol = split_a_list_at_zeros(current_solution)
	
	bound = bound_prob_vehicle_vehicle

	# To swap from
	non_empty_lists = [(idx+1) for idx, l in enumerate(sol) if len(l) > 0]
	veh_to_swap_from = choice(non_empty_lists)
	#print(vehicle_calls)
	#call_num = choice(list(vehicle_calls[veh_to_swap_from]))
	#print(f"{sol} {veh_to_swap_from}")
	call_num = choice(list(sol[veh_to_swap_from-1]))

	if random() > bound:
		end_range = num_vehicles
	else:
		end_range = num_vehicles-1

	#print(f"num of vehicles: {num_vehicles}")
	for veh_to_insert_into in range(1, end_range+1):
		if veh_to_insert_into == veh_to_swap_from:
			continue
	
		#veh_to_swap_from += 1
		#veh_to_insert_into += 1

		if call_num not in vehicle_calls[veh_to_insert_into]:
			continue

		solution_copy = current_solution.copy()
		#print(f"From {veh_to_swap_from} to {veh_to_insert_into}")
		_, new_sol = remove_call_from_array(problem, solution_copy, call_num, veh_to_swap_from)
		successfull, new_sol = greedy_insert_into_array(problem, new_sol, call_num, veh_to_insert_into)

		if successfull:
			new_cost = cost_function(new_sol, problem)
			if new_cost < best_cost:
				current_solution = new_sol.copy()
				best_cost = new_cost
	
	if len(original_sol) == len(current_solution):
		#print(f"=========\nChanged by greedyremoverandom\nOld: {original_sol}\nNew: {current_solution}")
		#print(original_sol==current_solution)
		return current_solution
	else:
		#print("return original")
		return original_sol

def alter_solution_greedy_insert_remove_highest_cost(problem: dict(), current_solution: List[int], helper_structure) -> List[int]:
	""" greedy insertion"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	original_sol = current_solution.copy()

	# Hyperparameters
	bound_prob_vehicle_vehicle = 0.8 # probability that dummy doesnt get reinserted

	logging.debug(f"Alter solution: greedy insert one vehicle, highest_cost_removal")

	q = randint(1,3)

	# ========================================
	best_cost = cost_function(current_solution, problem)
	for _ in range(q):
		veh_to_remove, call_to_remove = remove_highest_cost(problem, current_solution)
		
		if veh_to_remove == -1:
			continue

		sol = split_a_list_at_zeros(current_solution)
		vehicles_to_insert = [veh_idx for veh_idx in range(len(sol)-1) if veh_idx in vehicle_calls[veh_idx+1] and veh_to_remove-1 != veh_idx]

		if random() > bound_prob_vehicle_vehicle:
			vehicles_to_insert.append(len(sol)-1)
		
		for veh_to_insert_into in vehicles_to_insert:
			solution_copy = current_solution.copy()
			#print("=============")
			#print(f"Call: {call_to_remove}, veh: {veh_to_remove}->{veh_to_insert_into+1}")
			#print(f"Old: {solution_copy}")
			_, new_sol = remove_call_from_array(problem, solution_copy, call_to_remove, veh_to_remove)
			successfull, new_sol = greedy_insert_into_array(problem, new_sol, call_to_remove, veh_to_insert_into+1)
			#print(f"Success: {successfull}")
			#print(f"New: {new_sol}")

			if successfull:
				feasibility, _ = feasibility_check(new_sol, problem)
				new_cost = cost_function(new_sol, problem)
				if new_cost < best_cost:
					current_solution = new_sol.copy()
					best_cost = new_cost
	# ========================================
 
	if len(original_sol) == len(current_solution):
		#print(f"=========\nq: {q}, Changed by greedyremovehighestcost\nOld: {original_sol}\nNew: {current_solution}")
		#print(original_sol==current_solution)
		return current_solution
	else:
		#print("return original")
		return original_sol


def bla():
	# TODO regret insertion
	return 0

def alter_solution_placeholder5(problem: dict(), current_solution: List[int], helper_structure) -> List[int]:
	# TODO first possible insertion
	return current_solution

def local_search(problem: dict(), init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [1,2,3]):
	""" Local loops n-times over the neighbours of the currently best solution
		If the randomly chosen neighbour is better than the current one, and its feasible,
		it is the new solution to build neighbours from"""
	logging.info(f"Start local search with neighbour(s) {allowed_neighbours}")

	cost = cost_function(init_sol, problem)
	sol = init_sol
	orig_cost = cost

	for i in range(num_of_iterations):
		neighbourfunc_id = choice(allowed_neighbours)
		if neighbourfunc_id == 0:
			new_sol = alter_solution_1insert(problem, sol, 0.8)
		elif neighbourfunc_id == 1:
			new_sol = alter_solution_2exchange(problem, sol)
		else:
			new_sol = alter_solution_3exchange(problem, sol)

		feasiblity, _ = feasibility_check(new_sol, problem)
		if feasiblity:
			new_cost = cost_function(new_sol, problem)

			if new_cost < cost:
				cost = new_cost
				sol = new_sol
	
	improvement = round(100*(orig_cost-cost)/orig_cost, 2)
	logging.debug(f"Original cost: {orig_cost}")
	logging.debug(f"New cost: {cost}")
	logging.debug(f"Improvement: {improvement}%")

	return sol, cost, improvement

def simulated_annealing(problem: dict(), init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [1,2,3]):
	""" Simulated annealing algorithm as stated in the slides of Ahmed"""
	logging.info(f"Start simulated annealing with neighbour(s) {allowed_neighbours}")

	t_f = 0.1 # final temperature
	cost = cost_function(init_sol, problem)
	best_sol = init_sol
	inc_sol = init_sol
	best_cost = cost
	inc_cost = cost
	orig_cost = cost

	delta_w = list()

	w = 0
	while w < 100 or not delta_w:
		neighbourfunc_id = choice(allowed_neighbours)
		if neighbourfunc_id == 0:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 1:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		else:
			new_sol = alter_solution_3exchange(problem, inc_sol)

		feasiblity, _ = feasibility_check(new_sol, problem)
		if feasiblity:
			new_cost = cost_function(new_sol, problem)
			delta_e = new_cost - inc_cost
			if delta_e < 0:
				inc_sol = new_sol
				inc_cost = new_cost
				if inc_cost < best_cost:
					best_sol = inc_sol
					best_cost = inc_cost
			elif random() < 0.8:
				inc_sol = new_sol
				inc_cost = new_cost
				delta_w.append(delta_e)
		w += 1
	
	delta_avg = sum(delta_w)/len(delta_w)

	t_0 = (-delta_avg)/math.log(0.8)
	alpha = (t_f/t_0) ** (1/(num_of_iterations-w))
	t = t_0

	for i in range(num_of_iterations-w):
		neighbourfunc_id = choice(allowed_neighbours)
		if neighbourfunc_id == 0:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 1:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		else:
			new_sol = alter_solution_3exchange(problem, inc_sol)

		feasiblity, _ = feasibility_check(new_sol, problem)
		if feasiblity:
			new_cost = cost_function(new_sol, problem)
			delta_e = new_cost - inc_cost

			if delta_e < 0:
				inc_sol = new_sol
				inc_cost = new_cost
				if inc_cost < best_cost:
					best_sol = inc_sol
					best_cost = inc_cost
			elif random() < (math.e ** (-delta_e/t)):
				inc_sol = new_sol
				inc_cost = new_cost
		
		t = alpha * t

	improvement = round(100*(orig_cost-best_cost)/orig_cost, 2)
	logging.debug(f"Original cost: {orig_cost}")
	logging.debug(f"New cost: {best_cost}")
	logging.debug(f"Improvement: {improvement}%")

	return best_sol, best_cost, improvement

def improved_simulated_annealing(problem: dict(), init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [4, 5, 6], probabilities: list = [1/3, 1/3, 1/3]):
	""" Improved simulated annealing algorithm as stated in the slides of Ahmed for assignment 4"""
	logging.info(f"Start improved simulated annealing with neighbour(s) {allowed_neighbours}")

	t_f = 0.1 # final temperature
	cost = cost_function(init_sol, problem)
	
	best_sol = init_sol
	inc_sol = init_sol
	best_cost = cost
	inc_cost = cost
	orig_cost = cost

	delta_w = list()

	helper_structure = problem_to_helper_structure(problem, init_sol)

	w = 0
	while w < 100 or not delta_w:
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]
		# Note: The original numbers have been changed from [0, 1, 2] to [1, 2, 3]
		if neighbourfunc_id == 1:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 2:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		elif neighbourfunc_id == 3:
			new_sol = alter_solution_3exchange(problem, inc_sol)
		elif neighbourfunc_id == 4:
			new_sol = alter_solution_4kinsert(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 5:
			new_sol = alter_solution_regretk(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 6:
			new_sol = alter_solution_greedy_insert(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 7:
			new_sol = alter_solution_greedy_insert_one_vehicle(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 8:
			new_sol = alter_solution_placeholder5(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 9:
			new_sol = alter_solution_greedy_insert_remove_highest_cost(problem, inc_sol, helper_structure)

		feasiblity, _ = feasibility_check(new_sol, problem)

		changed = False
		if feasiblity:
			new_cost = cost_function(new_sol, problem)
			delta_e = new_cost - inc_cost

			if delta_e < 0:
				inc_sol = new_sol
				inc_cost = new_cost
				if inc_cost < best_cost:
					best_sol = inc_sol
					#print(f"Changed, cost: {best_cost}->{inc_cost}")
					best_cost = inc_cost
					changed = True
			else:
				if random() < 0.8:
					inc_sol = new_sol
					#print(f"legg til delta_w, cost: {inc_cost}->{new_cost}")
					inc_cost = new_cost
				delta_w.append(delta_e)
		w += 1
		#print(f"{best_sol}, nbfunc: {neighbourfunc_id}, {changed}")
	
	delta_avg = sum(delta_w)/len(delta_w)
	#print(f"delta_avg: {delta_avg}")
	#print(f"delta_w: {delta_w}")

	t_0 = (-delta_avg)/math.log(0.8)
	#print(f"t_0={t_0}, t_f={t_f}, num_it: {num_of_iterations}, w: {w}")
	alpha = (t_f/t_0) ** (1/(num_of_iterations-w))
	#print(f"Alpha: {alpha}")
	t = t_0

	for i in range(num_of_iterations-w):
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]
		# Note: The original numbers have been changed from [0, 1, 2] to [1, 2, 3]
		if neighbourfunc_id == 1:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 2:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		elif neighbourfunc_id == 3:
			new_sol = alter_solution_3exchange(problem, inc_sol)
		elif neighbourfunc_id == 4:
			new_sol = alter_solution_4kinsert(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 5:
			new_sol = alter_solution_regretk(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 6:
			new_sol = alter_solution_greedy_insert(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 7:
			new_sol = alter_solution_greedy_insert_one_vehicle(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 8:
			new_sol = alter_solution_placeholder5(problem, inc_sol, helper_structure)
		elif neighbourfunc_id == 9:
			new_sol = alter_solution_greedy_insert_remove_highest_cost(problem, inc_sol, helper_structure)

		feasiblity, _ = feasibility_check(new_sol, problem)
		changed = False
		if feasiblity:
			new_cost = cost_function(new_sol, problem)
			delta_e = new_cost - inc_cost

			if delta_e < 0:
				inc_sol = new_sol
				inc_cost = new_cost
				if inc_cost < best_cost:
					best_sol = inc_sol
					best_cost = inc_cost
					changed = True
			elif random() < (math.e ** (-delta_e/t)):
				inc_sol = new_sol
				inc_cost = new_cost
		#print(f"{best_sol}, nbfunc: {neighbourfunc_id}, {changed}")
		
		#print(f"t: {t}, alpha: {alpha}")
		t = alpha * t

	improvement = round(100*(orig_cost-best_cost)/orig_cost, 2)
	logging.debug(f"Original cost: {orig_cost}")
	logging.debug(f"New cost: {best_cost}")
	logging.debug(f"Improvement: {improvement}%")

	return best_sol, best_cost, improvement
	
def local_search_sim_annealing_latex(problem: dict(), init_sol: list(), num_of_iterations: int = 10000, num_of_rounds: int = 10, allowed_neighbours: list = [1,2,3], probabilities: list = [1/3, 1/3, 1/3], method:str = "ls"):
	""" Performs any sort of heuristic on a number of neighbours
		It runs n times and takes the average of all of it, also returning the time consumption
		It finally runs the \LaTeX methods to add a new solution to the table 
		to the report PDF and change the optimal solution"""

	if method == "ls":
		logging.debug("Start local search \LaTeX")
	elif method == "sa":
		logging.debug("Start simulated annealing \LaTeX")
	elif method == "isa":
		logging.debug("Start improved simulated annealing \LaTeX")
	
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	average_times = []
	best_cost = float('inf')
	best_solution = []
	seeds = []
	improvements = []
	average_objectives = []

	for round_nr in range(num_of_rounds):
		start_time = timer()
		new_seed = randint(0, 10**9)
		seed(new_seed)
		seeds.append(new_seed)

		if method == "ls":
			method_str = "Local Search"
			sol, cost, improvement = local_search(problem, init_sol, num_of_iterations, allowed_neighbours)
		elif method == "sa":
			method_str = "Simulated Annealing"
			sol, cost, improvement = simulated_annealing(problem, init_sol, num_of_iterations, allowed_neighbours)
		elif method == "isa":
			method_str = "Simulated Annealing"
			sol, cost, improvement = improved_simulated_annealing(problem, init_sol, num_of_iterations, allowed_neighbours, probabilities)
		if allowed_neighbours == [0]:
			method_str += "-1-insert"
		elif allowed_neighbours == [0,1]:
			method_str += "-2-exchange"
		elif allowed_neighbours == [0,1,2]:
			method_str += "-3-exchange"
		elif allowed_neighbours == [4,5,6]:
			if probabilities == [1/3, 1/3, 1/3]:
				method_str += "SA-new operators (equal weights)"
			else:
				method_str += "SA-new operators (tuned weights)"

		finish_time = timer()
		average_times.append(finish_time-start_time)
		improvements.append(improvement)
		average_objectives.append(cost)

		print(f"Cost: {cost}")
		if cost < best_cost:
			best_cost = cost
			best_solution = sol

	average_objective = round(sum(average_objectives) / len(average_objectives), 2)
	improvement = max(improvements)
	average_time = round(sum(average_times) / len(average_times), 2)

	latex_add_line(num_vehicles = num_vehicles, num_calls = num_calls, method = method_str, average_obj = average_objective, best_obj = best_cost, improvement = improvement, running_time = average_time)
	
	return num_vehicles, num_calls, best_solution, best_cost, seeds