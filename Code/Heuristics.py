from typing import List
from collections import defaultdict
import logging
from random import randint, randrange, random, choice, seed, choices, sample, shuffle
from timeit import default_timer as timer
import math
import bisect

from Utils import *

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

def alter_solution_4placeholdername(problem: dict(), current_solution: List[int]) -> List[int]:
	pass

	# Stuff to choose from
	"""
	removed_solution, to_remove = remove_random_call(current_solution, problem, number_to_remove)
	removed_solution, to_remove = remove_highest_cost_call(current_solution, problem, number_to_remove)
	removed_solution, to_remove = remove_dummy_call(current_solution, problem, number_to_remove)
	solution = insert_regretk(current_solution, problem, to_remove, 2)
	solution = insert_greedy(current_solution, problem, to_remove)
	solution = insert_back_to_dummy(current_solution, problem, to_remove)"""

def alter_solution_5placeholdername(problem: dict(), current_solution: List[int]) -> List[int]:
	pass

def alter_solution_6placeholdername(problem: dict(), current_solution: List[int]) -> List[int]:
	pass

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
			new_sol = alter_solution_4placeholdername(problem, inc_sol)
		elif neighbourfunc_id == 5:
			new_sol = alter_solution_5placeholdername(problem, inc_sol)
		elif neighbourfunc_id == 6:
			new_sol = alter_solution_6placeholdername(problem, inc_sol)	
		
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

	arr = dict()
	delta_avg = sum(delta_w)/len(delta_w)

	t_0 = (-delta_avg)/math.log(0.8)
	alpha = (t_f/t_0) ** (1/(num_of_iterations-w))
	t = t_0

	for i in range(num_of_iterations-w):
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]
		if neighbourfunc_id == 1:
			new_sol = alter_solution_1insert(problem, inc_sol, 0.8)
		elif neighbourfunc_id == 2:
			new_sol = alter_solution_2exchange(problem, inc_sol)
		elif neighbourfunc_id == 3:
			new_sol = alter_solution_3exchange(problem, inc_sol)
		elif neighbourfunc_id == 4:
			new_sol = alter_solution_4placeholdername(problem, inc_sol)
		elif neighbourfunc_id == 5:
			new_sol = alter_solution_5placeholdername(problem, inc_sol)
		elif neighbourfunc_id == 6:
			new_sol = alter_solution_6placeholdername(problem, inc_sol)

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

	#print(arr)
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

		if cost < best_cost:
			best_cost = cost
			best_solution = sol

	average_objective = round(sum(average_objectives) / len(average_objectives), 2)
	improvement = max(improvements)
	average_time = round(sum(average_times) / len(average_times), 2)

	latex_add_line(num_vehicles = num_vehicles, num_calls = num_calls, method = method_str, average_obj = average_objective, best_obj = best_cost, improvement = improvement, running_time = average_time)
	
	return num_vehicles, num_calls, best_solution, best_cost, seeds