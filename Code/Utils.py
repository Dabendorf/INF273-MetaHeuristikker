from typing import List
import numpy as np
from collections import defaultdict
import logging
import random
from timeit import default_timer as timer
from random import random, sample, choices
import math

from enum import Enum
class ReasonNotFeasible(Enum):
	call_in_vehicle_not_allowed = 1
	vehicle_overloaded = 2
	time_window_wrong = 3
	time_window_wrong_specific = 4

logger = logging.getLogger(__name__)

def load_problem(filename: str):
	"""
	Function which reads an input file into a datastructure

	:param filename: Address of the problem input file
	:return: Named tuple object of problem attributes
	"""
	logger.debug(f"Reading input file {filename}: Start")
	temp_vehicle_info = []
	temp_vehicle_call_list = []
	temp_call_info = []
	temp_travel_times = []
	temp_node_costs = []

	# Reading the file
	with open(filename) as f:
		# Read 1: number of nodes
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'number of nodes'")
		num_nodes = int(f.readline().strip())

		# Read 2: number of vehicles
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'number of vehicles'")
		num_vehicles = int(f.readline().strip())

		# Read 3: for each vehicle: idx, home node, starting time, capacity (4 columns)
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'for each vehicles (time, capacity)'")
		for i in range(num_vehicles):
			temp_vehicle_info.append(f.readline().strip().split(","))

		# Read 4: number of calls
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'number of calls'")
		num_calls = int(f.readline().strip())

		# Read 5: for each vehicle: idx, [list of possible calls] (2 columns)
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'for each vehicles (list of transportable calls)'")
		for i in range(num_vehicles):
			temp_vehicle_call_list.append(f.readline().strip().split(","))

		# Read 6: for each call: idx, origin_node, dest_node, size, ... (9 columns)
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'for each call'")
		for i in range(num_calls):
			temp_call_info.append(f.readline().strip().split(","))
		
		# Read 7: travel times and costs (5 columns)
		line = f.readline().strip()
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'travel times and costs'")

		line = f.readline()
		while not line.startswith("%"):
			temp_travel_times.append(line.strip().split(","))
			line = f.readline()

		# Read 8: node times and costs (6 columns), read until EOF
		if not line.startswith("%"):
			raise ValueError("Missing comment line 'node times and costs'")
		line = f.readline()
		while not line.startswith("%"):
			temp_node_costs.append(line.strip().split(","))
			line = f.readline()
	
	logger.debug(f"Reading input file {filename}: Finish")
	logger.debug(f"Converting input data: Start")
		
	# Travel times and costs (vehicle, origin_node, dest_node) = (travel_time, travel_cost)
	travel_times_costs = dict()
	for el in temp_travel_times:
		travel_times_costs[(int(el[0]), int(el[1]), int(el[2]))] = (int(el[3]), int(el[4]))

	# Node times and costs (vehicle, call) = (orig_time, orig_cost, dest_time, dest_cost)
	node_time_costs = dict()
	for el in temp_node_costs:
		node_time_costs[(int(el[0]), int(el[1]))] = (int(el[2]), int(el[3]), int(el[4]), int(el[5]))
	
	# Vehicle information 2D-List with [idx, home_node, starting_time, capacity]
	vehicle_info = np.array(temp_vehicle_info, dtype=np.int)

	# Call information 2D-List with [idx, origin_node, dest_node, size, cost_of_not_transporting, earliest_pickup_time, latest_pickup_time, earliest_delivery_time, latest_delivery_time]
	call_info = np.array(temp_call_info, dtype=np.int)

	# Dictionary of lists of calls per vehicle dict[idx] = list(call_numbers)
	vehicle_calls = dict()
	for el in temp_vehicle_call_list:
		vehicle_calls[int(el[0])] = set(map(int, el[1:]))

	# num_nodes			int 	number of nodes
	# num_vehicles 		int 	number of vehicles
	# num_calls			int		number of calls
	# travel_time_costs dict[(vehicle, origin_node, dest_node)] = (travel_time, travel_cost)	travel time and cost for each tuple (vehicle, start_node, dest_node)
	# node_time_costs	dict[(vehicle, call)] = (orig_time, orig_cost, dest_time, dest_cost) Node times and costs 
	# vehicle_info		2D-List with [idx, home_node, starting_time, capacity]	Vehicle information 
	# call_info			2D-List with [idx, origin_node, dest_node, size, cost_of_not_transporting, earliest_pickup_time, latest_pickup_time, earliest_delivery_time, latest_delivery_time]	Call information
	# vehicle_calls		dict[idx] = set(call_numbers)	Dictionary of set of calls per vehicle

	# Random probabilities
	lam = 0.2
	probabilities =  [math.e**(-lam*(x))-math.e**(-lam*(x+1)) for x in range(num_calls)]

	logger.debug(f"Converting input data: Finish")
	# return output as a dictionary
	output = {
		"num_nodes": num_nodes,
		"num_vehicles": num_vehicles,
		"num_calls": num_calls,
		"travel_time_cost": travel_times_costs,
		"node_time_cost": node_time_costs,
		"vehicle_info": vehicle_info,
		"call_info": call_info,
		"vehicle_calls": vehicle_calls,
		"prob": probabilities,
	}

	return output

def feasibility_check(solution: list(), problem: dict()):
	"""Checks if a solution is feasibile and if not what the reason for that is

	:param solution: The input solution of order of calls for each vehicle to the problem
	:param problem: The pickup and delivery problem dictionary
	:return: whether the problem is feasible and the reason for probable infeasibility
	"""
	logging.debug(f"Start feasibility check")
	logging.debug(f"Solution: {solution}")
	logging.debug(f"Problem keys: {problem.keys()}")

	num_vehicles = problem["num_vehicles"]
	vehicle_info = problem["vehicle_info"]
	vehicle_calls = problem["vehicle_calls"]
	call_info = problem["call_info"]
	travel_cost_dict = problem["travel_time_cost"]
	node_cost_dict = problem["node_time_cost"]

	reason_not_feasible = ""

	# Checks three conditions
	# (1) Check if calls and vehicles are compatible
	#sol_split_by_vehicle = split_a_list_at_zeros(solution)[0:num_vehicles]
	sol_split_by_vehicle = solution[0:num_vehicles]
	logging.debug(f"Solution split by vehicle: {sol_split_by_vehicle}")

	for veh_ind, l in enumerate(sol_split_by_vehicle):
		set_visited = set(l)
		set_allowed_to_visit = set(vehicle_calls[veh_ind+1])
		
		# if building set difference everything should disappear if set_visited only contains valid points
		# if not, the length > 1 and an illegal call was served
		if len(set_visited-set_allowed_to_visit) > 0:
			logging.debug(f"Solution not feasible - Vehicle served call without permission")
			reason_not_feasible = ReasonNotFeasible.call_in_vehicle_not_allowed
			return (True if reason_not_feasible == "" else False), reason_not_feasible

	# (2) Capacity of the vehicle
	for veh_ind, l in enumerate(sol_split_by_vehicle):
		size_available = vehicle_info[veh_ind][3]
		
		calls_visited = set()
		for call in l:
			if call in calls_visited:
				calls_visited.remove(call)
				size_available += call_info[call-1][3]
			else:
				calls_visited.add(call)
				size_available -= call_info[call-1][3]
				if size_available < 0:
					logging.debug(f"Solution not feasible - Vehicle {veh_ind+1} got overloaded")
					reason_not_feasible = ReasonNotFeasible.vehicle_overloaded
					return (True if reason_not_feasible == "" else False), reason_not_feasible

	# (3) Time windows at both nodes
	veh_times = list()
	
	# loop through all vehicles
	for veh_ind, l in enumerate(sol_split_by_vehicle):
		# Starting time of each vehicle
		curr_time = vehicle_info[veh_ind][2]

		# Only check feasibility if vehicle is not empty
		length_list = len(l)
		if length_list > 0:
			calls_visited = set()

			# Get home node
			home_node = vehicle_info[veh_ind][1]

			# First call number
			call_numb = l[0]
			# Information about first call number
			ci = call_info[call_numb-1]
			pickup_node = ci[1]

			goal_node = home_node

			# Go through all other nodes
			for i in range(0, length_list):
				start_node = goal_node

				call_numb = l[i]-1
				ci = call_info[call_numb]

				if call_numb+1 in calls_visited:
					goal_node = ci[2]
				else:
					goal_node = ci[1]

				next_travel_time = travel_cost_dict[(veh_ind+1, start_node, goal_node)][0]
				
				curr_time += next_travel_time

				# if already visited, delivery
				if call_numb+1 in calls_visited:
					calls_visited.remove(call_numb+1)
	
					lower_del, upper_del = ci[7:9]

					if curr_time > upper_del:
						logging.debug(f"Solution not feasible - Vehicle {veh_ind+1} came too late")
						reason_not_feasible = ReasonNotFeasible.time_window_wrong
						curr_time -= next_travel_time
						return (True if reason_not_feasible == "" else False), reason_not_feasible
					if curr_time < lower_del:
						curr_time = lower_del
					
					next_loading_time = node_cost_dict[(veh_ind+1, call_numb+1)][2]
					curr_time += next_loading_time

				# if not visited yet, pickup
				else:
					calls_visited.add(call_numb+1)

					lower_pickup, upper_pickup = ci[5:7]

					if curr_time > upper_pickup:
						logging.debug(f"Solution not feasible - Vehicle {veh_ind+1} came too late")
						reason_not_feasible = ReasonNotFeasible.time_window_wrong
						curr_time -= next_travel_time
						return (True if reason_not_feasible == "" else False), reason_not_feasible
					if curr_time < lower_pickup:
						curr_time = lower_pickup

					next_loading_time = node_cost_dict[(veh_ind+1, call_numb+1)][0]
					curr_time += next_loading_time


		# Remove later
		veh_times.append(curr_time)
	
	logging.debug(f"Feasible: {(True if reason_not_feasible == '' else False)}, Reason: {reason_not_feasible}")
	return (True if reason_not_feasible == "" else False), reason_not_feasible

def cost_function(solution: list(), problem: dict()):
	"""
	Function calculates the cost (not to confuse with time) of a solution
	This consists of transportation cost, origin and destination costs and cost of not transporting

	:param solution: the proposed solution for the order of calls in each vehicle
	:param problem: dictionary of problem data
	:return: Integer with costs
	"""
	logging.debug(f"Start cost function")
	logging.debug(f"Solution: {solution}")
	logging.debug(f"Problem keys: {problem.keys()}")

	num_vehicles = problem["num_vehicles"]
	call_info = problem["call_info"]
	travel_cost_dict = problem["travel_time_cost"]
	node_cost_dict = problem["node_time_cost"]
	vehicle_info = problem["vehicle_info"]

	not_transport_cost = 0
	sum_travel_cost = 0
	sum_node_cost = 0

	# Start calculate not transported costs
	dummy_list = set(solution[-1])
	logging.debug(f"Ports not visited: {dummy_list}")

	for not_vis in dummy_list:
		not_transport_cost += call_info[not_vis-1][4]
	# Finish calculate not transported costs
	
	logging.debug(f"Cost not transporting: {not_transport_cost}")

	sol_split_by_vehicle = solution[0:num_vehicles]
	logging.debug(f"Solution split by vehicle: {sol_split_by_vehicle}")

	# Loop for costs of nodes and transport
	for veh_ind, l in enumerate(sol_split_by_vehicle):
		set_visited = list(set(l))
		for call_ind in set_visited:
			# Nodes
			call_cost_list = node_cost_dict[(veh_ind+1, call_ind)]
			sum_node_cost += (call_cost_list[1] + call_cost_list[3])

		# Transport (edges)
		length_list = len(l)
		if length_list > 0:
			calls_visited = set()
			home_node = vehicle_info[veh_ind][1]
			call_numb = l[0]-1
			calls_visited.add(call_numb)
			ci = call_info[call_numb]
			start_node = ci[1]

			sum_travel_cost += travel_cost_dict[(veh_ind+1, home_node, start_node)][1]
			
			for i in range(1, length_list):
				call_numb = l[i]-1
				if call_numb in calls_visited:
					calls_visited.remove(call_numb)
					ci = call_info[call_numb]
					goal_node = ci[2]
				else:
					calls_visited.add(call_numb)
					ci = call_info[call_numb]
					goal_node = ci[1]
				sum_travel_cost += travel_cost_dict[(veh_ind+1, start_node, goal_node)][1]
				start_node = goal_node

	logging.debug(f"Cost of nodes: {sum_node_cost}")
	logging.debug(f"Cost of travel: {sum_travel_cost}")

	total_cost = not_transport_cost + sum_travel_cost + sum_node_cost
	logging.debug(f"Total costs: {total_cost}")
	return total_cost

def split_a_list_at_zeros(k: list()):
	""" Function which takes as argument a valid solution and
		breaks it down into vehicle sublists by splitting at the zeros"""
	output_list = list()
	while k:
		try:
			ind = k.index(0)
			output_list.append(k[:ind])
			k = k[ind+1:]
		except ValueError:
			output_list.append(k)
			k = []

	return output_list

def random_solution(problem: dict()):
	""" Random solution generator generates valid but not necessarily feasible solution
		The current version is medium good
		It gives one call to each vehicle and outsources the rest of it"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	call_list = list(range(1,num_calls+1))
	random.shuffle(call_list)

	overall_list = list()
	for v in range(num_vehicles):
		overall_list += ([call_list[v]]+[call_list[v]]+[0])
	overall_list += [val for val in call_list[num_vehicles:] for _ in range(2)]
	logging.debug(f"Creating a random solution {overall_list}")

	return overall_list

def initial_solution_old(problem: dict()) -> List[int]:
	""" This function generates an initial solution
		where only calls are in the dummy vehicle"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	sol = [0] * num_vehicles
	sol += [val for val in list(range(1,num_calls+1)) for _ in (0, 1)]
	logging.debug(f"Generate inital dummy solution: {sol}")

	return sol

def initial_solution(problem: dict()) -> List[int]:
	""" This function generates an initial solution
		where only calls are in the dummy vehicle
		Format: One list per vehicle """
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	sol = list()
	
	for i in range(num_vehicles):
		sol.append(list())

	dummy = [[val for val in list(range(1,num_calls+1)) for _ in (0, 1)]]
	sol.extend(dummy)
	logging.debug(f"Generate inital dummy solution: {sol}")

	return sol

def solution_to_ahmed_output(sol: List[List[int]]) -> List[int]:
	""" Takes understandable solution format in 2D list and
		transforms it into inflexible output format in 1D list
	"""
	output = list()
	for i in range(len(sol)-1):
		output.extend(sol[i])
		output.append(0)
	
	output.extend(sol[-1])
	return output

def blind_random_search(problem: dict(), num_of_iterations: int = 10000):
	""" This method does a blind search which generates
		a bunch of random solutions and returns the best of it
		
		It returns four values:
		feasibility: if one of the solutions was feasible (must be true)
		sol: the best solution
		cost: the cost of the best feasible solution
		counter: the number of generated feasible solutions"""

	logging.debug("Start blind search")

	counter = 0
	
	# Initial solution
	sol = initial_solution(problem=problem)

	feasiblity, _ = feasibility_check(sol, problem)
	if feasiblity:
		counter += 1
		cost = cost_function(sol, problem) 
	else:
		cost = float('inf')

	for i in range(num_of_iterations):
		new_sol = random_solution(problem)
		new_feasiblity, _ = feasibility_check(new_sol, problem)

		if new_feasiblity:
			counter += 1
			new_cost = cost_function(new_sol, problem)
			feasiblity = new_feasiblity
			if new_cost < cost:
				sol = new_sol
				cost = new_cost
	logging.debug(f"Generate final solution: {sol}")
	if not feasiblity:
		logging.error(f"Generate non feasible solution: {sol}")

	return feasiblity, sol, cost, counter

def blind_random_search_latex(problem: dict(), num_of_iterations: int = 10000):
	""" This method does a blind search which generates
		a bunch of random solutions and returns the best of it
		This one is suited for the output in terms of INF273 conditions
		
		It returns four values:
		feasibility: if one of the solutions was feasible (must be true)
		sol: the best solution
		cost: the cost of the best feasible solution
		counter: the number of generated feasible solutions
		average_objective: the average cost of all feasible solutions
		cost: the cost of the best solution
		improvement: how much best solution improved from original one"""

	logging.debug("Start blind search \LaTeX")
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	counter = 0
	
	# Initial solution
	sol = [0] * num_vehicles
	sol += [val for val in list(range(1,num_calls+1)) for _ in (0, 1)]
	logging.debug(f"Generate inital dummy solution: {sol}")

	feasiblity, _ = feasibility_check(sol, problem)
	all_costs = []
	if feasiblity:
		counter += 1
		cost = cost_function(sol, problem) 
		all_costs.append(cost)
		original_cost = cost
	else:
		cost = float('inf')
		original_cost = cost

	for i in range(num_of_iterations):
		new_sol = random_solution(problem)
		new_feasiblity, _ = feasibility_check(new_sol, problem)

		if new_feasiblity:
			counter += 1
			new_cost = cost_function(new_sol, problem)
			all_costs.append(new_cost)
			feasiblity = new_feasiblity
			if new_cost < cost:
				sol = new_sol
				cost = new_cost
	logging.debug(f"Generate final solution: {sol}")

	average_objective = round(sum(all_costs) / len(all_costs), 2)
	improvement = round(100*(original_cost-cost)/original_cost, 2)
	if not feasiblity:
		logging.error(f"Generate non feasible solution: {sol}")

	return feasiblity, sol, cost, counter, average_objective, improvement

def blind_search_latex_generator(problem: dict(), num_of_iterations: int = 10000, num_of_blind_searchs: int = 10):
	""" This method runs the blind search several times
		at generates beautiful LaTeX code"""

	logging.debug("Start to write a LaTeX table of random solutions")
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	with open("solution_table.tex", "a") as f:
		f.write("\\begin{table}[ht]\n")
		f.write("\\centering\n")
		f.write(f"\\caption{{Call\_{num_calls}\_Vehicle\_{num_vehicles}}}\n")
		f.write(f"\\label{{tab:call{num_calls}vehicle{num_vehicles}}}\n")
		f.write("\\begin{tabular}{|r|r|r|r|r|}\n")
		f.write("Method & Average objective & Best objective & Improvement (\%) & Running time \\\\\n")
		f.write("\hline\n")
		
		average_times = []
		best_cost = float('inf')
		best_solution = []
		seeds = []
		improvements = []
		average_objectives = []

		for i in range(num_of_blind_searchs):
			start_time = timer()
			seed = random.randint(0, 10**9)
			random.seed(seed)
			seeds.append(seed)

			feasiblity, sol, cost, counter, average_objective, improvement = blind_random_search_latex(problem=problem, num_of_iterations=num_of_iterations)
			if not feasiblity:
				logging.error(f"Generate non feasible solution: {sol}")
			
			finish_time = timer()
			average_times.append(finish_time-start_time)
			improvements.append(improvement)
			average_objectives.append(average_objective)

			if cost < best_cost:
				best_cost = cost
				best_solution = sol

		f.write(f"Random search & {round(sum(average_objectives) / len(average_objectives), 2):.2f} & {best_cost} & {round(sum(improvements) / len(improvements), 2):.2f}\% & {round(sum(average_times) / len(average_times), 2):.2f}s\\\\\n")
		
		f.write(f"\end{{tabular}}%Call\_{num_calls}\_Vehicle\_{num_vehicles}\n")
		f.write("\end{table}\n")
		f.write(f"\\begin{{lstlisting}}[label={{lst:call{num_calls}vehicle{num_vehicles}}},caption=Optimal solution call\_{num_calls}\_vehicle\_{num_vehicles}]\n")
		if len(best_solution) < 150:
			f.write(f"sol = {best_solution}\n")
		else:
			f.write(f"sol = {str(best_solution[0:150])[:-1]},\n")
			f.write(f"      {str(best_solution[150:])[1:]}\n")
		f.write(f"seeds = {seeds}\n")
		f.write(f"\end{{lstlisting}}%Call\_{num_calls}\_Vehicle\_{num_vehicles}\n")
		f.write("\clearpage")
		f.write("\n\n\n")

		logging.info(f"Finished to write a LaTeX table of random solutions for file call\_{num_calls}\_vehicle\_{num_vehicles}")

def latex_add_line(num_vehicles: int, num_calls: int, method:str, average_obj: float, best_obj: int, improvement: float, running_time: float):
	""" This method adds another element into the LaTeX results table"""

	logging.debug("Start to write a new line into LaTeX table")
	path_file = "solution_table.tex"

	with open(path_file, "r") as f:
		contents = f.readlines()

	try:
		idx = contents.index(f"\end{{tabular}}%Call\_{num_calls}\_Vehicle\_{num_vehicles}\n")
	except:
		logging.error(f"ValueError: there is no table for the file Call_{num_calls}_Vehicle_{num_vehicles}")
	

	new_line = f"{method} & {average_obj:.2f} & {best_obj} & {improvement:.2f}\% & {running_time:.2f}s\\\\\n"
	contents.insert(idx, new_line)

	with open(path_file, "w") as f:
		contents = "".join(contents)
		f.write(contents)
	logging.debug("Finish to write a new line into LaTeX table")

def latex_replace_line(num_vehicles: int, num_calls: int, best_solution, seeds):
	""" This method replaces the optimal solution and the seeds with a new one"""

	logging.debug("Start to replace optimal solution in table")
	path_file = "solution_table.tex"

	with open(path_file, "r") as f:
		contents = f.readlines()

	try:
		idx = contents.index(f"\end{{lstlisting}}%Call\_{num_calls}\_Vehicle\_{num_vehicles}\n")
	except:
		logging.error(f"\end{{lstlisting}}%Call\_{num_calls}\_Vehicle\_{num_vehicles}\n")
		logging.error(f"ValueError: there is no table for the file Call_{num_calls}_Vehicle_{num_vehicles}")
		exit(0)

	if len(best_solution) < 150:
		sol_line = f"sol = {best_solution}\n"
		contents[idx-2] = sol_line
	else:
		sol_line1 = f"sol = {str(best_solution[0:150])[:-1]},\n"
		sol_line2 = f"      {str(best_solution[150:])[1:]}\n"
		contents[idx-3] = sol_line1
		contents[idx-2] = sol_line2

	seeds_line = f"seeds = {seeds}\n"
	contents[idx-1] = seeds_line

	with open(path_file, "w") as f:
		contents = "".join(contents)
		f.write(contents)
	logging.debug("Finish to replace optimal solution in table")

def merge_vehice_lists(splitted_solution: list()):
	overall_list = list()
	for i in splitted_solution:
		overall_list.extend(i)
		overall_list.append(0)
	
	return overall_list[:-1]

def feasibility_helper(solution: list(), problem: dict(), vehicle_num: int, call_num_to_check=None):
	""" This is a helper function which checks if the solution for one specific vehicle is feasible or not
	It is a shorter version of the long feasibility function

	:param solution: The input solution of order of calls for one vehicle
	:param problem: The pickup and delivery problem dictionary
	:param vehicle_num: Exact vehicle_num between [1, num_vehicles]
	:param call_num_to_check: Optional, if set, checks if this specific call is reason for wrong time window
	:return: whether the solution is feasible and the reason for probable infeasibility
	"""
	logging.debug(f"Start helper feasibility function")
	vehicle_info = problem["vehicle_info"]
	call_info = problem["call_info"]
	travel_cost_dict = problem["travel_time_cost"]
	node_cost_dict = problem["node_time_cost"]
	num_vehicles = problem["num_vehicles"]

	if vehicle_num > num_vehicles:
		return True, ""

	reason_not_feasible = ""

	# (2) Capacity of the vehicle
	veh_ind = vehicle_num-1
	l = solution
	size_available = vehicle_info[veh_ind][3]
	
	calls_visited = set()
	for call in l:
		if call in calls_visited:
			calls_visited.remove(call)
			size_available += call_info[call-1][3]
		else:
			calls_visited.add(call)
			size_available -= call_info[call-1][3]
			if size_available < 0:
				logging.debug(f"Solution not feasible - Vehicle {veh_ind+1} got overloaded")
				reason_not_feasible = ReasonNotFeasible.vehicle_overloaded
				return (True if reason_not_feasible == "" else False), reason_not_feasible

	# (3) Time windows at both nodes
	
	# loop through all vehicles
	# Starting time of each vehicle
	curr_time = vehicle_info[veh_ind][2]

	# Only check feasibility if vehicle is not empty
	length_list = len(l)
	if length_list > 0:
		calls_visited = set()

		# Get home node
		home_node = vehicle_info[veh_ind][1]

		# First call number
		call_numb = l[0]
		# Information about first call number
		ci = call_info[call_numb-1]
		pickup_node = ci[1]

		goal_node = home_node

		# Go through all other nodes
		for i in range(0, length_list):
			start_node = goal_node

			call_numb = l[i]-1
			ci = call_info[call_numb]

			if call_numb+1 in calls_visited:
				goal_node = ci[2]
			else:
				goal_node = ci[1]

			next_travel_time = travel_cost_dict[(veh_ind+1, start_node, goal_node)][0]
			
			curr_time += next_travel_time

			# if already visited, delivery
			if call_numb+1 in calls_visited:
				calls_visited.remove(call_numb+1)

				lower_del, upper_del = ci[7:9]

				if curr_time > upper_del:
					logging.debug(f"Solution not feasible - Vehicle {veh_ind+1} came too late")
					reason_not_feasible = ReasonNotFeasible.time_window_wrong
					if call_num_to_check != None:
						if veh_ind+1 == call_num_to_check:
							reason_not_feasible = ReasonNotFeasible.time_window_wrong_specific

					curr_time -= next_travel_time
					return (True if reason_not_feasible == "" else False), reason_not_feasible
				if curr_time < lower_del:
					curr_time = lower_del
				
				next_loading_time = node_cost_dict[(veh_ind+1, call_numb+1)][2]
				curr_time += next_loading_time

			# if not visited yet, pickup
			else:
				calls_visited.add(call_numb+1)

				lower_pickup, upper_pickup = ci[5:7]

				if curr_time > upper_pickup:
					logging.debug(f"Solution not feasible - Vehicle {veh_ind+1} came too late")
					reason_not_feasible = ReasonNotFeasible.time_window_wrong
					if call_num_to_check != None:
						if veh_ind+1 == call_num_to_check:
							reason_not_feasible = ReasonNotFeasible.time_window_wrong_specific

					curr_time -= next_travel_time
					return (True if reason_not_feasible == "" else False), reason_not_feasible
				if curr_time < lower_pickup:
					curr_time = lower_pickup

				next_loading_time = node_cost_dict[(veh_ind+1, call_numb+1)][0]
				curr_time += next_loading_time

	
	logging.debug(f"Feasible: {(True if reason_not_feasible == '' else False)}, Reason: {reason_not_feasible}")
	return (True if reason_not_feasible == "" else False), reason_not_feasible

def cost_helper(solution: list(), problem: dict(), vehicle_num: int):
	"""
	Function calculates the cost (not to confuse with time) of a vehicle
	This consists of transportation cost, origin and destination costs

	:param solution: the proposed solution for the order of calls in one vehicle
	:param problem: dictionary of problem data
	:param vehicle_num: Exact vehicle_num between [1, num_vehicles]
	:return: Integer with costs
	"""
	logging.debug(f"Start cost function helper for vehicle {vehicle_num}")
	logging.debug(f"Solution: {solution}")
	logging.debug(f"Problem keys: {problem.keys()}")

	call_info = problem["call_info"]
	travel_cost_dict = problem["travel_time_cost"]
	node_cost_dict = problem["node_time_cost"]
	vehicle_info = problem["vehicle_info"]
	num_vehicles = problem["num_vehicles"]

	sum_travel_cost = 0
	sum_node_cost = 0

	if vehicle_num > num_vehicles:
		not_transport_cost = 0
		dummy_list = set(solution)

		for not_vis in dummy_list:
			not_transport_cost += call_info[not_vis-1][4]
		return not_transport_cost
	else:
		# Loop for costs of nodes and transport
		veh_ind = vehicle_num-1
		l = solution

		set_visited = list(set(l))

		for call_ind in set_visited:
			# Nodes
			call_cost_list = node_cost_dict[(veh_ind+1, call_ind)]
			sum_node_cost += (call_cost_list[1] + call_cost_list[3])

		# Transport (edges)
		length_list = len(l)
		if length_list > 0:
			calls_visited = set()
			home_node = vehicle_info[veh_ind][1]
			call_numb = l[0]-1
			calls_visited.add(call_numb)
			ci = call_info[call_numb]
			start_node = ci[1]

			sum_travel_cost += travel_cost_dict[(veh_ind+1, home_node, start_node)][1]
			
			for i in range(1, length_list):
				call_numb = l[i]-1
				if call_numb in calls_visited:
					calls_visited.remove(call_numb)
					ci = call_info[call_numb]
					goal_node = ci[2]
				else:
					calls_visited.add(call_numb)
					ci = call_info[call_numb]
					goal_node = ci[1]
				sum_travel_cost += travel_cost_dict[(veh_ind+1, start_node, goal_node)][1]
				start_node = goal_node

		logging.debug(f"Cost of nodes: {sum_node_cost}")
		logging.debug(f"Cost of travel: {sum_travel_cost}")

		total_cost = sum_travel_cost + sum_node_cost
		logging.debug(f"Total costs: {total_cost}")
		return total_cost

def cost_helper_transport_only(solution: list(), problem: dict(), vehicle_num: int):
	"""
	Function calculates the transport cost (not to confuse with time) of a vehicle
	It ignores node and "not transport" cost

	:param solution: the proposed solution for the order of calls in one vehicle
	:param problem: dictionary of problem data
	:param vehicle_num: Exact vehicle_num between [1, num_vehicles]
	:return: Integer with costs
	"""
	logging.debug(f"Start cost function helper for vehicle {vehicle_num}")
	logging.debug(f"Solution: {solution}")

	call_info = problem["call_info"]
	travel_cost_dict = problem["travel_time_cost"]
	node_cost_dict = problem["node_time_cost"]
	vehicle_info = problem["vehicle_info"]
	num_vehicles = problem["num_vehicles"]

	sum_travel_cost = 0

	if vehicle_num > num_vehicles:
		not_transport_cost = 0
		dummy_list = set(solution)

		for not_vis in dummy_list:
			not_transport_cost += call_info[not_vis-1][4]
		return not_transport_cost
	else:
		# Loop for costs of nodes and transport
		veh_ind = vehicle_num-1
		l = solution

		# Transport (edges)
		length_list = len(l)
		if length_list > 0:
			calls_visited = set()
			home_node = vehicle_info[veh_ind][1]
			call_numb = l[0]-1
			calls_visited.add(call_numb)
			ci = call_info[call_numb]
			start_node = ci[1]

			sum_travel_cost += travel_cost_dict[(veh_ind+1, home_node, start_node)][1]
			
			for i in range(1, length_list):
				call_numb = l[i]-1
				if call_numb in calls_visited:
					calls_visited.remove(call_numb)
					ci = call_info[call_numb]
					goal_node = ci[2]
				else:
					calls_visited.add(call_numb)
					ci = call_info[call_numb]
					goal_node = ci[1]
				sum_travel_cost += travel_cost_dict[(veh_ind+1, start_node, goal_node)][1]
				start_node = goal_node

		logging.debug(f"Cost of travel: {sum_travel_cost}")

		logging.debug(f"Total costs: {sum_travel_cost}")
		return sum_travel_cost

def remove_random_call(solution: list(), problem: dict(), number_to_remove: int):
	""" Removes n calls from the call list
		Returns: (new solution, list of removed calls) """

	num_calls = problem["num_calls"]

	to_remove = set(sample(range(1, num_calls), number_to_remove))

	new_solution = [[x for x in inner if x not in to_remove] for inner in solution]

	logging.debug(f"new solution: {new_solution}, removed: {to_remove}; Remove {number_to_remove} random calls")
	return new_solution, to_remove

def remove_highest_cost_call(solution: list(), problem: dict(), number_to_remove: int):
	""" Removes the n highest cost calls (not from dummy)
		Its not always taking out the highest cost, but giving those a higher probability (diversification)
		Returns: (new solution, list of removed calls) """
	
	probs = problem["prob"]

	cost_of_removal_dict = dict()
	lookup_which_vehicle = dict()

	# Remove dummy
	all_but_dummy = solution[:-1]

	# Go through all vehicles having calls
	for veh_idx, veh in enumerate(all_but_dummy):
		if len(veh) > 0:
			init_cost = cost_helper(veh, problem, veh_idx+1)
			calls_in_vehicle = set(veh)
			
			# Go through all calls and find difference between vehicle with and without that call
			for call in calls_in_vehicle:
				cost_without_call = cost_helper([x for x in veh if x != call], problem, veh_idx+1)
				cost_of_removal_dict[call] = init_cost-cost_without_call
				lookup_which_vehicle[call] = veh_idx+1
	
	highest_cost_calls = sorted(cost_of_removal_dict, key=cost_of_removal_dict.get, reverse=True)
	len_calls = len(highest_cost_calls)

	if len_calls > 0:
		number_to_remove = min(len_calls, number_to_remove)

		probs = probs[:len_calls]
		# Normalise the weights to sum to 1
		weights = [w/sum(probs) for w in probs]

		# Random choice based on exponential probability
		to_remove = set(np.random.choice(highest_cost_calls, size=number_to_remove, replace=False, p=weights))

		new_solution = [[x for x in inner if x not in to_remove] for inner in solution]

		logging.debug(f"new solution: {new_solution}, removed: {to_remove}; Remove {number_to_remove} random calls")
		return new_solution, to_remove
	else:
		# Return dummy if there are no calls (only initial solution)
		return remove_dummy_call(solution, problem, number_to_remove)

def remove_dummy_call(solution: list(), problem: dict(), number_to_remove: int):
	""" Removes n calls from the call list, but only from the dummy
		Returns: (new solution, list of removed calls) """

	calls_in_dummy = set(solution[-1])
	num_calls_in_dummy = len(calls_in_dummy)
	number_to_remove = min(num_calls_in_dummy, number_to_remove)

	to_remove = set(sample(calls_in_dummy, number_to_remove))

	solution[-1] = [x for x in solution[-1] if x not in to_remove]

	logging.debug(f"new solution: {solution}, removed: {to_remove}; Remove {number_to_remove} dummy calls")
	return solution, to_remove

def insert_regretk(solution: List[List[int]], problem: dict(), calls_to_insert: List[int], k: int) -> List[List[int]]:
	""" It takes n calls and looks for its best k insertion positions
		It then puts in those first who have the higher regret value
		:param solution: The original full solution array
		:param problem: The problem representation
		:param calls_to_insert: A set of all calls to be inserted
		:param k: The regret value

		return: The new solution
	"""
	pass

def insert_greedy(solution: list(), problem: dict(), calls_to_insert: List[int]):
	""" It takes n calls and inserts each of them greedy
		:param solution: The original full solution array
		:param problem: The problem representation
		:param calls_to_insert: A set of all calls to be inserted

		return: The new solution
	"""

	logging.debug(f"Start insert greedy")

	num_vehicles = problem["num_vehicles"]
	vehicle_calls = problem["vehicle_calls"]

	
	output_sol = solution.copy()
	for call_num in calls_to_insert:
		call_solution = output_sol.copy()
		#print(f"New call to insert: {call_num}, current solution: {call_solution}")
		# Search for all vehicles which can take that call
		#  and veh_to_remove != (veh_idx+1)
		vehicles_insertable = [(veh_idx+1) for veh_idx in range(num_vehicles) if (call_num) in vehicle_calls[veh_idx+1]]
		#print(f"call_num: {call_num}, allowed_veh: {vehicles_insertable}")

		best_cost = float("inf")
		success_once = False

		for veh_num in vehicles_insertable:
			orig_sol_one_veh = call_solution[veh_num-1]
			temp_sol_one_veh, successful = greedy_insert_one_call_one_vehicle(orig_sol_one_veh, problem, call_num, veh_num)
			#print(f"Try insert {call_num} in {veh_num}: {temp_sol_one_veh}, {successful}")

			if orig_sol_one_veh != temp_sol_one_veh and successful:
				success_once = True
				temp_sol = call_solution.copy()
				temp_sol[veh_num-1] = temp_sol_one_veh
				temp_cost = cost_function(temp_sol, problem)
				if temp_cost < best_cost:
					call_solution = temp_sol.copy()
					best_cost = temp_cost
		output_sol = call_solution
		if not success_once:
			output_sol[-1].insert(0, call_num)
			output_sol[-1].insert(0, call_num)
	
	return output_sol

def greedy_insert_one_call_one_vehicle(vehicle_solution: List[List[int]], problem: dict(), call_to_insert: List[int], vehicle_to_insert: List[int]):
	""" It takes one call and one vehicle and inserts it greedily
		:param vehicle_solution: The original solution of that vehicle
		:param problem: The problem representation
		:param call_to_insert: The call num to insert [1, num_calls]
		:param vehicle_to_insert: The vehicle num to insert [1, num_vehicles]

		return: The new solution
	"""

	logging.debug(f"Start insert greedy for one vehicle")

	num_vehicles = problem["num_vehicles"]
	
	len_call_list = len(vehicle_solution)
	best_cost = float("inf")
	output_sol = vehicle_solution.copy()

	# if dummy, just insert it
	if vehicle_to_insert > num_vehicles:
		vehicle_solution.insert(call_to_insert)
		vehicle_solution.insert(call_to_insert)
	else:
		#print(f"Start greedy one insert call {call_to_insert}, vehicle: {vehicle_to_insert}")
		# Each call must be placed to times, so there are for loops for both of them
		continue_outer_search = True
		for insert_idx_1 in range(len_call_list+1):
			if not continue_outer_search:
				break

			temp_call_list_1 = vehicle_solution.copy()
			temp_call_list_1.insert(insert_idx_1, call_to_insert)
			is_feas_1, reason_not_feas_1 = feasibility_helper(temp_call_list_1, problem, vehicle_to_insert, call_num_to_check=call_to_insert)

			#print(f"ind1: {insert_idx_1}, success: {is_feas_1}, {reason_not_feas_1}, {temp_call_list_1}")
			if is_feas_1 or reason_not_feas_1 == ReasonNotFeasible.vehicle_overloaded:
				for insert_idx_2 in range(insert_idx_1, len_call_list+2):
					temp_call_list_2 = temp_call_list_1.copy()
					temp_call_list_2.insert(insert_idx_2, call_to_insert)

					is_feas_2, reason_not_feas_2 = feasibility_helper(temp_call_list_2, problem, vehicle_to_insert, call_num_to_check=call_to_insert)
					#print(f"ind2: {insert_idx_2}, success: {is_feas_2}, {reason_not_feas_2}, {temp_call_list_2}")
					if is_feas_2:
						new_cost = cost_helper_transport_only(temp_call_list_2, problem, vehicle_to_insert)

						if new_cost < best_cost:
							if random() < 0.8 or best_cost == float("inf"):
								best_cost = new_cost
								output_sol = temp_call_list_2
					elif reason_not_feas_2 == ReasonNotFeasible.time_window_wrong_specific:
						break

			elif reason_not_feas_1 == ReasonNotFeasible.time_window_wrong_specific:
				continue_outer_search = False

	return output_sol, True if best_cost != float("inf") else False