from typing import List
import numpy as np
from collections import defaultdict
import logging
from random import randint, randrange, random, choice
import numpy as np
from timeit import default_timer as timer
import math

from numpy import choose
from Utils import split_a_list_at_zeros, cost_function, feasibility_check

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

def local_search(problem: dict(), init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [1,2,3]):
	""" Local loops n-times over the neighbours of the currently best solution
		If the randomly chosen neighbour is better than the current one, and its feasible,
		it is the new solution to build neighbours from"""
	logging.info(f"Start local search with neighbour(s) {allowed_neighbours}")

	cost = cost_function(init_sol, problem)
	sol = init_sol
	orig_cost = cost
	print(f"Original cost: {orig_cost}")

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
	
	print(cost)
	improvement = round(100*(orig_cost-cost)/orig_cost, 2)
	print(f"Improvement: {improvement}%")

def simulated_annealing(problem: dict(), init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [1,2,3]):
	""" """

	logging.info(f"Start simulated annealing with neighbour(s) {allowed_neighbours}")
	t_f = 0.1 # final temperature
	cost = cost_function(init_sol, problem)
	best_sol = init_sol
	inc_sol = init_sol
	best_cost = cost
	inc_cost = cost
	orig_cost = cost
	print(f"Original cost: {orig_cost}")

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
	#print(f"len delta_w: {len(delta_w)}")
	#print(f"Delta avg: {delta_avg}")

	t_0 = (-delta_avg)/math.log(0.8)
	#print(f"t_0: {t_0}")
	#print(f"Iterations for second loop: {num_of_iterations-w}")
	alpha = (t_f/t_0) ** (1/(num_of_iterations-w))
	#print(f"Alpha: {alpha}")
	t = t_0
	#print(f"Start temp: {t}")

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

	print(best_cost)
	improvement = round(100*(orig_cost-best_cost)/orig_cost, 2)
	print(f"Improvement: {improvement}%")
	