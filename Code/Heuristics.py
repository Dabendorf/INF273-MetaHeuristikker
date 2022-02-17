from typing import List
import numpy as np
from collections import defaultdict
import logging
from random import randint, randrange, random, choice
import numpy as np
from timeit import default_timer as timer
from Utils import split_a_list_at_zeros

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

def local_search(problem: dict(), num_of_iterations: int = 10000):
	""" """
	pass

def simulated_annealing(problem: dict(), num_of_iterations: int = 10000):
	""" """
	pass