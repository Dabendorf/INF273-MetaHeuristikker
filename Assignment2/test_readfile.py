from unittest import TestCase

from simplejson import load
from Utils import load_problem
import pytest
import random
	
class ReadProblem(TestCase):
	pytest.problem_file = load_problem("Data/Call_7_Vehicle_3.txt")

	def test_length_problem_dict(self):
		assert len(pytest.problem_file) == 8

	def test_length_information_in_dict(self):
		counter = 0
		for k, v in pytest.problem_file.items():
			if type(v) is int:
				counter += 1
			else:
				counter += len(v)

		assert (counter+9) == sum(1 for line in open("Data/Call_7_Vehicle_3.txt"))

	def test_datatype_num_nodes(self):
		assert type(pytest.problem_file["num_nodes"]) == int

	def test_datatype_num_vehicles(self):
		assert type(pytest.problem_file["num_vehicles"]) == int

	def test_datatype_num_calls(self):
		assert type(pytest.problem_file["num_calls"]) == int

	def test_columns_travel_time_cost(self):
		assert len(random.choice(list(pytest.problem_file["travel_time_cost"].values()))) == 2

	def test_columns_node_time_costs(self):
		assert len(random.choice(list(pytest.problem_file["node_time_cost"].values()))) == 4

	def test_columns_vehicle_info(self):
		assert len(pytest.problem_file["vehicle_info"][0]) == 4

	def test_columns_call_info(self):
		assert len(pytest.problem_file["call_info"][0]) == 9

	def test_columns_vehicle_calls(self):
		assert type(random.choice(list(pytest.problem_file["vehicle_calls"].values()))) == list
