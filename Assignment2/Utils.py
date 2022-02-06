from cProfile import label
import enum
import numpy as np
from collections import namedtuple
import logging
from itertools import groupby

from sqlalchemy import false

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
			break

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
					break

	# (3) Time windows at both nodes
	
	logging.debug(f"Feasible: {(True if reason_not_feasible == '' else False)}, Reason: {reason_not_feasible}")
	return (True if reason_not_feasible == "" else False), reason_not_feasible

	"""

	solution = np.append(solution, [0])
	zero_index = np.array(np.where(solution == 0)[0], dtype=int)
	feasibility = True
	tempidx = 0
	c = 'Feasible'
	for i in range(num_vehicles):
		currentVPlan = solution[tempidx:zero_index[i]]
		currentVPlan = currentVPlan - 1
		no_double_call_on_vehicle = len(currentVPlan)
		tempidx = zero_index[i] + 1
		if no_double_call_on_vehicle > 0:

			if not np.all(vessel_cargo[i, currentVPlan]):
				feasibility = False
				c = 'incompatible vessel and cargo'
				break
			else:
				load_size = 0
				current_time = 0
				sortRout = np.sort(currentVPlan)
				I = np.argsort(currentVPlan)
				indx = np.argsort(I)
				load_size -= cargo[sortRout, 2]
				load_size[::2] = cargo[sortRout[::2], 2]
				load_size = load_size[indx]
				if np.any(vessel_capacity[i] - np.cumsum(load_size) < 0):
					feasibility = False
					c = 'Capacity exceeded'
					break
				time_windows = np.zeros((2, no_double_call_on_vehicle))
				time_windows[0] = cargo[sortRout, 6]
				time_windows[0, ::2] = cargo[sortRout[::2], 4]
				time_windows[1] = cargo[sortRout, 7]
				time_windows[1, ::2] = cargo[sortRout[::2], 5]

				time_windows = time_windows[:, indx]

				port_index = cargo[sortRout, 1].astype(int)
				port_index[::2] = cargo[sortRout[::2], 0]
				port_index = port_index[indx] - 1

				lu_time = unloading_time[i, sortRout]
				lu_time[::2] = loading_time[i, sortRout[::2]]
				lu_time = lu_time[indx]
				diag = travel_time[i, port_index[:-1], port_index[1:]]
				first_visit_time = first_travel_time[i, int(
					cargo[currentVPlan[0], 0] - 1)]

				route_travel_time = np.hstack((first_visit_time, diag.flatten()))

				arrive_time = np.zeros(no_double_call_on_vehicle)
				for j in range(no_double_call_on_vehicle):
					arrive_time[j] = np.max(
						(current_time + route_travel_time[j], time_windows[0, j]))
					if arrive_time[j] > time_windows[1, j]:
						feasibility = False
						c = 'Time window exceeded at call {}'.format(j)
						break
					current_time = arrive_time[j] + lu_time[j]

	return feasibility, c"""


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
	""" Splits a list into sublists by positions of zeros
		Function sponsored by Stackoverflow:
		https://stackoverflow.com/questions/71007348/split-list-into-several-lists-at-specific-values
		"""
	gr = groupby(k,  lambda a: a==0)
	l = [[] if a else [*b] for a,b in gr]
	return [ a for idx,a in enumerate(l) if idx in (0,len(l)) or a]