from typing import List
from collections import defaultdict
import logging
from random import randint, randrange, random, choice, seed, choices
from timeit import default_timer as timer
import math
from copy import deepcopy

from Utils import split_a_list_at_zeros, insert_greedy, insert_regretk, cost_function, feasibility_check, remove_dummy_call, remove_random_call, remove_highest_cost_call, latex_add_line, solution_to_hashable_tuple_2d

logger = logging.getLogger(__name__)

def alter_solution_1insert(problem: dict(), current_solution: List[int], bound_prob_vehicle_vehicle: float) -> List[int]:
	""" 1insert takes a call from one vehicle (including dummy) and puts it into another one
		:param problem: Problem dictionary
		:param current_solution: Full solution list
		:param bound_prob_vehicle_vehicle: Probability that there is no reinsert into the dummy"""
	num_vehicles = problem["num_vehicles"]
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
			#if count_call_iterations == 10:
				#logging.debug("Did not swap anything, nothing to get swapped found")
				#return current_solution

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
	""" 2exchange swaps one call from one vehicle with another call from another vehicle
		:param problem: Problem dictionary
		:param current_solution: Full solution list
	"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	vehicle_calls = problem["vehicle_calls"]

	logging.debug(f"Alter solution: 2-exchange")
	log_message = ""

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
				logging.debug("Did not find anything, swap two random calls")
				call1 = randint(1, num_calls)
				call2 = randint(1, num_calls)
				indices1 = [i for i, x in enumerate(current_solution) if x == call1]
				indices2 = [i for i, x in enumerate(current_solution) if x == call2]
				current_solution[indices1[0]] = call2
				current_solution[indices1[1]] = call2
				current_solution[indices2[0]] = call1
				current_solution[indices2[1]] = call1
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
				logging.debug("Did not find anything, swap two random calls")
				call1 = randint(1, num_calls)
				call2 = randint(1, num_calls)
				indices1 = [i for i, x in enumerate(current_solution) if x == call1]
				indices2 = [i for i, x in enumerate(current_solution) if x == call2]
				current_solution[indices1[0]] = call2
				current_solution[indices1[1]] = call2
				current_solution[indices2[0]] = call1
				current_solution[indices2[1]] = call1
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
	""" 3exchange swaps one call each from three different vehicles with each other
		:param problem: Problem dictionary
		:param current_solution: Full solution list
	"""
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
				logging.debug("Did not find anything, swap two random calls")
				call1 = randint(1, num_calls)
				call2 = randint(1, num_calls)
				call3 = randint(1, num_calls)
				indices1 = [i for i, x in enumerate(current_solution) if x == call1]
				indices2 = [i for i, x in enumerate(current_solution) if x == call2]
				indices3 = [i for i, x in enumerate(current_solution) if x == call2]
				current_solution[indices1[0]] = call2
				current_solution[indices1[1]] = call2
				current_solution[indices2[0]] = call3
				current_solution[indices2[1]] = call3
				current_solution[indices3[0]] = call1
				current_solution[indices3[1]] = call1
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
				logging.debug("Did not find anything, swap two random calls")
				call1 = randint(1, num_calls)
				call2 = randint(1, num_calls)
				call3 = randint(1, num_calls)
				indices1 = [i for i, x in enumerate(current_solution) if x == call1]
				indices2 = [i for i, x in enumerate(current_solution) if x == call2]
				indices3 = [i for i, x in enumerate(current_solution) if x == call2]
				current_solution[indices1[0]] = call2
				current_solution[indices1[1]] = call2
				current_solution[indices2[0]] = call3
				current_solution[indices2[1]] = call3
				current_solution[indices3[0]] = call1
				current_solution[indices3[1]] = call1
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

def alter_solution_4steven(problem: dict(), current_solution: List[int]) -> List[int]:
	""" A combination of removing n random calls and inserting those greedily"""
	""" Functions to choose from
	removed_solution, to_remove, removed_from = remove_random_call(current_solution, problem, number_to_remove)
	removed_solution, to_remove, removed_from = remove_highest_cost_call(current_solution, problem, number_to_remove)
	removed_solution, to_remove, removed_from = remove_dummy_call(current_solution, problem, number_to_remove)
	solution = insert_regretk(removed_solution, problem, to_remove, 2)
	solution = insert_greedy(removed_solution, problem, to_remove)
	# solution = insert_back_to_dummy(removed_solution, problem, to_remove)"""
	removed_solution, to_remove, removed_from = remove_random_call(current_solution, problem, randint(1,3))
	solution = insert_greedy(removed_solution, problem, to_remove, removed_from)
	return solution

def alter_solution_5jackie(problem: dict(), current_solution: List[int]) -> List[int]:
	""" A combination of removing n highest cost calls and inserting them with regretk"""

	removed_solution, to_remove, removed_from = remove_highest_cost_call(current_solution, problem, randint(1,3))
	solution = insert_regretk(removed_solution, problem, to_remove, removed_from, 2)
	return solution

def alter_solution_6sebastian(problem: dict(), current_solution: List[int]) -> List[int]:
	""" A combination of removing n dummy calls and inserting them greedily"""

	removed_solution, to_remove, removed_from = remove_dummy_call(current_solution, problem, randint(1,3))
	solution = insert_greedy(removed_solution, problem, to_remove, removed_from)
	return solution

def alter_solution_7steinar(problem: dict(), current_solution: List[int]) -> List[int]:
	""" A combination of removing n random calls and inserting them with regretk"""

	removed_solution, to_remove, removed_from = remove_random_call(current_solution, problem, randint(1,3))
	solution = insert_regretk(removed_solution, problem, to_remove, removed_from, 2)
	return solution

def alter_solution_8stian(problem: dict(), current_solution: List[int]) -> List[int]:
	""" A combination of removing n highest cost calls and inserting them greedily"""

	removed_solution, to_remove, removed_from = remove_highest_cost_call(current_solution, problem, randint(1,3))
	solution = insert_greedy(removed_solution, problem, to_remove, removed_from)
	return solution

def alter_solution_9karina(problem: dict(), current_solution: List[int]) -> List[int]:
	""" A combination of removing n dummy calls and inserting them with regretk"""

	removed_solution, to_remove, removed_from = remove_dummy_call(current_solution, problem, randint(1,3))
	solution = insert_regretk(removed_solution, problem, to_remove, removed_from, 2)
	return solution

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
		"""if neighbourfunc_id == 0:
			new_sol = alter_solution_1insert(problem, sol, 0.8)
		elif neighbourfunc_id == 1:
			new_sol = alter_solution_2exchange(problem, sol)
		else:
			new_sol = alter_solution_3exchange(problem, sol)"""
		if neighbourfunc_id == 1:
			new_sol = alter_solution_1insert(problem, sol, 0.8)
		elif neighbourfunc_id == 2:
			new_sol = alter_solution_2exchange(problem, sol)
		elif neighbourfunc_id == 3:
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
	logging.info("Finished this run")
	logging.info(f"Best cost: {best_cost}")
	logging.info(f"Best sol: {best_sol}")

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

	w = 0
	while w < 100 or not delta_w:
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]
		if neighbourfunc_id == 1:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 2:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		elif neighbourfunc_id == 3:
			new_sol = alter_solution_3exchange(problem, inc_sol)
		elif neighbourfunc_id == 4:
			new_sol = alter_solution_4steven(problem, inc_sol)
		elif neighbourfunc_id == 5:
			new_sol = alter_solution_5jackie(problem, inc_sol)
		elif neighbourfunc_id == 6:
			new_sol = alter_solution_6sebastian(problem, inc_sol)
		elif neighbourfunc_id == 7:
			new_sol = alter_solution_7steinar(problem, inc_sol)
		elif neighbourfunc_id == 8:
			new_sol = alter_solution_8stian(problem, inc_sol)
		elif neighbourfunc_id == 9:
			new_sol = alter_solution_9karina(problem, inc_sol)	
		
		"""print(f"New sol: {new_sol}, neighbour {neighbourfunc_id}")
		if len(solution_to_ahmed_output(new_sol))!=17:
			print("ERROR")
			print(len(solution_to_ahmed_output(new_sol)))"""
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
			else:
				if random() < 0.8:
					inc_sol = new_sol
					inc_cost = new_cost
				delta_w.append(delta_e)
		w += 1
		#print(new_sol, neighbourfunc_id)

	logging.info(f"Finished warmup")

	arr = dict()
	delta_avg = sum(delta_w)/len(delta_w)

	t_0 = (-delta_avg)/math.log(0.8)
	alpha = (t_f/t_0) ** (1/(num_of_iterations-w))
	t = t_0

	for i in range(num_of_iterations-w):
		if i%1000==0:
			logging.info(f"Iteration num: {i}")

		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]
		if neighbourfunc_id == 1:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 2:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		elif neighbourfunc_id == 3:
			new_sol = alter_solution_3exchange(problem, inc_sol)
		elif neighbourfunc_id == 4:
			new_sol = alter_solution_4steven(problem, inc_sol)
		elif neighbourfunc_id == 5:
			new_sol = alter_solution_5jackie(problem, inc_sol)
		elif neighbourfunc_id == 6:
			new_sol = alter_solution_6sebastian(problem, inc_sol)
		elif neighbourfunc_id == 7:
			new_sol = alter_solution_7steinar(problem, inc_sol)
		elif neighbourfunc_id == 8:
			new_sol = alter_solution_8stian(problem, inc_sol)
		elif neighbourfunc_id == 9:
			new_sol = alter_solution_9karina(problem, inc_sol)

		"""print(f"New sol: {new_sol}, neighbour {neighbourfunc_id}")
		if len(solution_to_ahmed_output(new_sol))!=17:
			print("ERROR2")
			print(len(solution_to_ahmed_output(new_sol)))"""
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

			else:
				p = math.e ** (-delta_e/t)
				if random() < p:
					inc_sol = new_sol
					inc_cost = new_cost
					arr[i] = p

		t = alpha * t
		#print(new_sol, neighbourfunc_id)

	#print(arr)
	improvement = round(100*(orig_cost-best_cost)/orig_cost, 2)
	logging.debug(f"Original cost: {orig_cost}")
	logging.debug(f"New cost: {best_cost}")
	logging.debug(f"Improvement: {improvement}%")

	return best_sol, best_cost, improvement

def adaptive_algorithm(problem: dict(), init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [4, 5, 6, 7, 8, 9], file_num = None, statistics=False):
	""" Adaptive algorithm inspired from simulated annealing and Ahmeds slides 12-20"""
	logging.info(f"Start adaptive algorithm with neighbour(s) {allowed_neighbours}")

	if statistics:
		import matplotlib.pyplot as plt
		from matplotlib.pyplot import figure

	# Dictionary of past probabilities
	prob_hist = defaultdict(lambda: list())
	prob_hist["y"].append(0)

	# Best solution (starts as initial)
	best_sol = init_sol
	s = init_sol.copy()

	cost = cost_function(init_sol, problem)
	best_cost = cost
	cost_s = cost
	iterations_since_best_found = 0
	last_iteration_found_best = 0

	# Save original cost
	orig_cost = cost

	# Dictionary of weights
	r = 0.2
	weights = dict()
	probabilities = list()
	for neighbour in allowed_neighbours:
		weight_val = 1/len(allowed_neighbours)
		weights[neighbour] = weight_val
		probabilities.append(weight_val)
		prob_hist[f"x{neighbour}"].append(weight_val)
	score_sums = defaultdict(lambda: 0)
	neighbour_used_counter = defaultdict(lambda: 0)

	# Found solutions
	found_sol = set()

	w = 0
	while w < num_of_iterations:
		if w%200 == 0:
			# Initialise set of neighbours not yet used
			not_used_yet = set(allowed_neighbours)

		if w%1000 == 0:
			logging.info(f"Iteration num: {w}")

		new_score_val = 0
		# +1 found a new solution not explored yet
		# +2 found better than current
		# +4 found new best

		if iterations_since_best_found > 100:
			s, cost_s, is_new_best = escape_algorithm(problem=problem, current_solution=s, allowed_neighbours=[4, 7], best_sol_cost=best_cost, cost_s=cost_s) # alternate operator
			# update best solution TODO 
			# current_solution, new_cost, False

			if is_new_best:
				best_sol = deepcopy(s)
				best_cost = cost_s
				last_iteration_found_best = w

			iterations_since_best_found = 0
		
		s2 = deepcopy(s)

		# Choose a neighbour function
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]

		# Use functions not used yet
		if w%200 > 150:
			if len(not_used_yet) > 0:
				neighbourfunc_id = choice(list(not_used_yet))
		
		neighbour_used_counter[neighbourfunc_id] += 1
		not_used_yet = not_used_yet.difference({neighbourfunc_id})

		# Apply neighbouring function
		if neighbourfunc_id == 1:
			s2 = alter_solution_1insert(problem, s2, 0.8)
		elif neighbourfunc_id == 2:
			s2 = alter_solution_2exchange(problem, s2)
		elif neighbourfunc_id == 3:
			s2 = alter_solution_3exchange(problem, s2)
		elif neighbourfunc_id == 4:
			s2 = alter_solution_4steven(problem, s2)
		elif neighbourfunc_id == 5:
			s2 = alter_solution_5jackie(problem, s2)
		elif neighbourfunc_id == 6:
			s2 = alter_solution_6sebastian(problem, s2)
		elif neighbourfunc_id == 7:
			s2 = alter_solution_7steinar(problem, s2)
		elif neighbourfunc_id == 8:
			s2 = alter_solution_8stian(problem, s2)
		elif neighbourfunc_id == 9:
			s2 = alter_solution_9karina(problem, s2)

		feasiblity, _ = feasibility_check(s2, problem)

		updated_value = False
		if feasiblity:
			new_cost = cost_function(s2, problem)

			if new_cost < best_cost:
				new_score_val = 4
				best_sol = s2
				best_cost = new_cost
				updated_value = True
				s = deepcopy(s2)
				cost_s = new_cost
				last_iteration_found_best = w
			
			elif new_cost < cost_s:
				new_score_val = 2
				s = deepcopy(s2)
				cost_s = new_cost
			
			elif random() < 0.2:
				s = deepcopy(s2)
				cost_s = new_cost
			
			hashed_sol = solution_to_hashable_tuple_2d(s2)
			if hashed_sol not in found_sol:
				new_score_val = 1
				found_sol.add(hashed_sol)

		if updated_value:
			iterations_since_best_found = 0
		else:
			iterations_since_best_found += 1
		
		w += 1
		# Update scores
		score_sums[neighbourfunc_id] += new_score_val
		if w%200 == 0:
			# update_parameters
			probabilities = []
			#print(f"Scores: {score_sums}")
			#print(f"Used: {neighbour_used_counter}")
			for neighbour in allowed_neighbours:
				new_weight = weights[neighbour] * (1-r) + r * (score_sums[neighbour]/neighbour_used_counter[neighbour])
				weights[neighbour] = new_weight
				score_sums[neighbour] = 0
				neighbour_used_counter[neighbour] = 0
				probabilities.append(new_weight)
			prob_hist["y"].append(w)

			sum_prob = sum(probabilities)
			for idx, el in enumerate(probabilities):
				probabilities[idx] = el/sum_prob
				prob_hist[f"x{idx+4}"].append(el/sum_prob)
			#print(f"Probabilities: {probabilities}")
			
			logging.debug(f"New weights: {probabilities}")

	if statistics:
		plt.figure(figsize=(20, 10))
		
		plt.axvline(x=last_iteration_found_best, color='b', label="best")
		y = prob_hist["y"]
		for k, v in prob_hist.items():
			if k != "y":
				label = k[1:]
				plt.plot(y, v, label = label)

		plt.legend()
		plt.savefig(f"./tempdata/weights{file_num}.png")
	logging.info(f"Last iteration with new best: {last_iteration_found_best}")
	
	improvement = round(100*(orig_cost-best_cost)/orig_cost, 2)
	logging.info(f"Final probabilities: {list(map(lambda x: round(x, ndigits=2), probabilities))}")
	logging.debug(f"Original cost: {orig_cost}")
	logging.debug(f"New cost: {best_cost}")
	logging.debug(f"Improvement: {improvement}%")

	return best_sol, best_cost, improvement

def escape_algorithm(problem: dict(), current_solution, allowed_neighbours, best_sol_cost, cost_s, num_iterations=20):
	""" This is the escape algorithm to get out of a local minimum"""
	found_new_feasible_solution = False
	iteration_num = 0
	probabilities = [1] * len(allowed_neighbours)
	
	while iteration_num < num_iterations or not found_new_feasible_solution:
		iteration_num += 1

		# Choose a neighbour function
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]

		# Apply neighbouring function
		if neighbourfunc_id == 1:
			s2 = alter_solution_1insert(problem, current_solution, 0.8)
		elif neighbourfunc_id == 2:
			s2 = alter_solution_2exchange(problem, current_solution)
		elif neighbourfunc_id == 3:
			s2 = alter_solution_3exchange(problem, current_solution)
		elif neighbourfunc_id == 4:
			s2 = alter_solution_4steven(problem, current_solution)
		elif neighbourfunc_id == 5:
			s2 = alter_solution_5jackie(problem, current_solution)
		elif neighbourfunc_id == 6:
			s2 = alter_solution_6sebastian(problem, current_solution)
		elif neighbourfunc_id == 7:
			s2 = alter_solution_7steinar(problem, current_solution)
		elif neighbourfunc_id == 8:
			s2 = alter_solution_8stian(problem, current_solution)
		elif neighbourfunc_id == 9:
			s2 = alter_solution_9karina(problem, current_solution)

		feasiblity, _ = feasibility_check(s2, problem)

		if feasiblity:
			new_cost = cost_function(s2, problem)
			current_solution = s2
			found_new_feasible_solution = True

			if new_cost < best_sol_cost:
				return current_solution, new_cost, True

	return current_solution, new_cost, False

def local_search_sim_annealing_latex(problem: dict(), init_sol: list(), num_of_iterations: int = 10000, num_of_rounds: int = 10, allowed_neighbours: list = [1,2,3], probabilities: list = [1/3, 1/3, 1/3], method:str = "aa", file_num=None, statistics=False):
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
	elif method == "aa":
		logging.debug("Start adaptive algorithm \LaTeX")
	
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	average_times = []
	best_cost = float('inf')
	best_solution = []
	seeds = []
	improvements = []
	average_objectives = []

	for round_nr in range(num_of_rounds):
		logging.info(f"Round number: {round_nr}")
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
			sol, cost, improvement = improved_simulated_annealing(problem, init_sol, num_of_iterations, allowed_neighbours, probabilities)
			if len(set(probabilities)) == 1:
				method_str = "SA-new operators (equal weights)"
			else:
				method_str = "SA-new operators (tuned weights)"
		elif method == "aa":
			method_str = "Adaptive Algorithm"
			sol, cost, improvement = adaptive_algorithm(problem, init_sol, num_of_iterations, allowed_neighbours, file_num, statistics=statistics)
		if allowed_neighbours == [0]:
			method_str += "-1-insert"
		elif allowed_neighbours == [0,1]:
			method_str += "-2-exchange"
		elif allowed_neighbours == [0,1,2]:
			method_str += "-3-exchange"

		finish_time = timer()
		average_times.append(finish_time-start_time)
		improvements.append(improvement)
		average_objectives.append(cost)

		if cost < best_cost:
			best_cost = cost
			best_solution = sol
		#print(cost)
		logging.info("Finished this run")
		logging.info(f"Best cost: {cost}")
		logging.info(f"Best sol: {sol}")

	average_objective = round(sum(average_objectives) / len(average_objectives), 2)
	improvement = max(improvements)
	average_time = round(sum(average_times) / len(average_times), 2)
	logging.info(f"Average cost: {average_objective}")

	latex_add_line(num_vehicles = num_vehicles, num_calls = num_calls, method = method_str, average_obj = average_objective, best_obj = best_cost, improvement = improvement, running_time = average_time)
	
	return num_vehicles, num_calls, best_solution, best_cost, seeds