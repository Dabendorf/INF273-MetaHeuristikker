from typing import List
import numpy as np
from collections import defaultdict
import logging
import random
from timeit import default_timer as timer
from itertools import chain

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
	sol_split_by_vehicle = split_a_list_at_zeros(solution)[0:num_vehicles]
	logging.debug(f"Solution split by vehicle: {sol_split_by_vehicle}")

	for veh_ind, l in enumerate(sol_split_by_vehicle):
		set_visited = set(l)
		set_allowed_to_visit = set(vehicle_calls[veh_ind+1])
		
		# if building set difference everything should disappear if set_visited only contains valid points
		# if not, the length > 1 and an illegal call was served
		if len(set_visited-set_allowed_to_visit) > 0:
			logging.debug(f"Solution not feasible - Vehicle served call without permission")
			reason_not_feasible = "Incompatible call and vehicle"
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
					reason_not_feasible = "Vehicle got overloaded"
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
						reason_not_feasible = "Vehicle came too late"
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
						reason_not_feasible = "Vehicle came too late"
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
	rev_sol = solution[::-1]
	ind_last_null = rev_sol.index(0)
	not_visited = set(rev_sol[:ind_last_null])
	logging.debug(f"Ports not visited: {not_visited}")

	for not_vis in not_visited:
		not_transport_cost += call_info[not_vis-1][4]
	# Finish calculate not transported costs
	logging.debug(f"Cost not transporting: {not_transport_cost}")

	sol_split_by_vehicle = split_a_list_at_zeros(solution)[0:num_vehicles]
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

def initial_solution(problem: dict()) -> List[int]:
	""" This function generates an initial solution
		where only calls are in the dummy vehicle"""
	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	sol = [0] * num_vehicles
	sol += [val for val in list(range(1,num_calls+1)) for _ in (0, 1)]
	logging.debug(f"Generate inital dummy solution: {sol}")

	return sol

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
		print(f"\end{{lstlisting}}%Call\_{num_calls}\_Vehicle\_{num_vehicles}\n")
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

def problem_to_helper_structure(problem: dict(), sol):
	""" This function takes a problem data structure and 
		outputs a helper data strcture to better insert information into it"""

	logging.debug("Start problem to helper structure method")

	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]

	print(problem.keys())
	print(f"Initial solution: {sol}")

	# in which vehicle is a call
	# init everything as dummy vehicle
	lookup_call_in_vehicle = [None]

	latest_arrival_time = list(list())

	arrival_info = dict()

	for call_num in range(num_calls):
		lookup_call_in_vehicle.append(num_vehicles+1)
		#latest_arrival_time.append((problem["call_info"][call_num][6], str(call_num+1)+"a"))
		#latest_arrival_time.append((problem["call_info"][call_num][8], str(call_num+1)+"b"))
	for vehicle in range(num_vehicles):
		veh_info = problem["vehicle_info"][vehicle]
		latest_arrival_time.append([(veh_info[2], "start", veh_info[1])])

	#latest_arrival_time.sort(reverse=True)
	print(f"Lookup Call->Vehicle: {lookup_call_in_vehicle}")
	print(f"Latest_arrival_time: {latest_arrival_time}")
	print(f"Arrival information: {arrival_info}")

	return [lookup_call_in_vehicle, latest_arrival_time, arrival_info]

def insert_call_into_array(problem: dict(), sol, helper_structure, call_num, vehicle_num):
	""" """

	logging.debug("Start problem to helper structure method")

	num_vehicles = problem["num_vehicles"]
	num_calls = problem["num_calls"]
	[lookup_call_in_vehicle, latest_arrival_time, arrival_info] = helper_structure

	# Split the vehicles and get the specific vehicle to insert into
	sol_split_by_vehicle = split_a_list_at_zeros(sol)
	call_list_vehicle = sol_split_by_vehicle[vehicle_num-1]
	print(call_list_vehicle)
	print(sol_split_by_vehicle)

	# If vehicle is empty, just insert the call two times
	if len(call_list_vehicle)==0:
		call_list_vehicle.append(call_num)
		call_list_vehicle.append(call_num)
		# Update the helper information
		# TODO
		lookup_call_in_vehicle[call_num] = vehicle_num
		#latest_arrival_time (time, 'call_num', 'node{a,b}')
		#arrival_info
	else:
		# Find correct insertion position
		# TODO
		# Update the helper information
		# TODO

		# Placeholder, to remove TODO
		call_list_vehicle.append(call_num)
		call_list_vehicle.append(call_num)
	
	# Remerge list and return the list and the helper structure
	sol_split_by_vehicle[vehicle_num-1] = call_list_vehicle
	return merge_vehice_lists(sol_split_by_vehicle), [lookup_call_in_vehicle, latest_arrival_time, arrival_info]

def remove_call_from_array(problem: dict(), sol, helper_structure, call_num, vehicle_num):
	pass

def merge_vehice_lists(splitted_solution: list()):
	overall_list = list()
	for i in splitted_solution:
		overall_list.extend(i)
		overall_list.append(0)
	
	return overall_list[:-1]
