from unittest import TestCase

from Utils import initial_solution, load_problem, feasibility_check, cost_function, split_a_list_at_zeros
import pytest
import random
	
class SolutionValidation(TestCase):
	pytest.paths_testfiles = ["Data/Call_7_Vehicle_3.txt", "Data/Call_18_Vehicle_5.txt", "Data/Call_35_Vehicle_7.txt", "Data/Call_80_Vehicle_20.txt", "Data/Call_130_Vehicle_40.txt", "Data/Call_300_Vehicle_90.txt"]

	def test_costs_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = split_a_list_at_zeros([0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6])
		cost = cost_function(sol, pytest.problem_file)

		assert cost == 1871372

	def test_costs_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = split_a_list_at_zeros([4, 4, 0, 8, 8, 0, 18, 18, 0, 12, 12, 0, 1, 1, 0, 11, 11, 16, 16, 2, 2, 13, 13, 15, 15, 10, 10, 7, 7, 5, 5, 14, 14, 3, 3, 9, 9, 6, 6, 17, 17])
		cost = cost_function(sol, pytest.problem_file)

		assert cost == 6229511

	def test_costs_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = split_a_list_at_zeros([28, 28, 0, 32, 32, 0, 26, 26, 0, 1, 1, 0, 31, 31, 0, 16, 16, 0, 12, 12, 0, 29, 29, 34, 34, 15, 15, 22, 22, 27, 27, 4, 4, 20, 20, 23, 23, 3, 3, 8, 8, 33, 33, 24, 24, 21, 21, 6, 6, 25, 25, 5, 5, 14, 14, 19, 19, 10, 10, 9, 9, 30, 30, 13, 13, 35, 35, 18, 18, 7, 7, 2, 2, 17, 17, 11, 11])
		cost = cost_function(sol, pytest.problem_file)

		assert cost == 14548954

	def test_costs_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = split_a_list_at_zeros([37, 37, 0, 40, 40, 0, 41, 41, 0, 32, 32, 0, 67, 67, 0, 1, 1, 0, 73, 73, 0, 56, 56, 0, 55, 55, 0, 21, 21, 0, 72, 72, 0, 46, 46, 0, 58, 58, 0, 31, 31, 0, 6, 6, 0, 17, 17, 0, 30, 30, 0, 79, 79, 0, 34, 34, 0, 35, 35, 0, 68, 68, 29, 29, 14, 14, 52, 52, 43, 43, 80, 80, 10, 10, 78, 78, 2, 2, 9, 9, 77, 77, 15, 15, 45, 45, 39, 39, 23, 23, 63, 63, 64, 64, 16, 16, 44, 44, 28, 28, 24, 24, 19, 19, 62, 62, 70, 70, 38, 38, 3, 3, 36, 36, 5, 5, 50, 50, 76, 76, 49, 49, 60, 60, 27, 27, 42, 42, 25, 25, 53, 53, 33, 33, 47, 47, 8, 8, 51, 51, 69, 69, 20, 20, 13, 13, 74, 74, 75, 75, 48, 48, 18, 18, 66, 66, 11, 11, 71, 71, 26, 26, 59, 59, 7, 7, 65, 65, 57, 57, 4, 4, 61, 61, 54, 54, 22, 22, 12, 12])
		cost = cost_function(sol, pytest.problem_file)

		assert cost == 37804910

	"""def test_costs_file4(self):
		# Note: no feasible solution found yet
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value

	def test_costs_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = []
		cost = cost_function(sol, pytest.problem_file)

		assert cost == value"""

	def test_feasible_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = split_a_list_at_zeros([5, 5, 0, 7, 7, 0, 1, 1, 0, 4, 4, 3, 3, 6, 6, 2, 2])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = split_a_list_at_zeros([4, 4, 0, 8, 8, 0, 18, 18, 0, 12, 12, 0, 1, 1, 0, 11, 11, 16, 16, 2, 2, 13, 13, 15, 15, 10, 10, 7, 7, 5, 5, 14, 14, 3, 3, 9, 9, 6, 6, 17, 17])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = split_a_list_at_zeros([28, 28, 0, 32, 32, 0, 26, 26, 0, 1, 1, 0, 31, 31, 0, 16, 16, 0, 12, 12, 0, 29, 29, 34, 34, 15, 15, 22, 22, 27, 27, 4, 4, 20, 20, 23, 23, 3, 3, 8, 8, 33, 33, 24, 24, 21, 21, 6, 6, 25, 25, 5, 5, 14, 14, 19, 19, 10, 10, 9, 9, 30, 30, 13, 13, 35, 35, 18, 18, 7, 7, 2, 2, 17, 17, 11, 11])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = split_a_list_at_zeros([37, 37, 0, 40, 40, 0, 41, 41, 0, 32, 32, 0, 67, 67, 0, 1, 1, 0, 73, 73, 0, 56, 56, 0, 55, 55, 0, 21, 21, 0, 72, 72, 0, 46, 46, 0, 58, 58, 0, 31, 31, 0, 6, 6, 0, 17, 17, 0, 30, 30, 0, 79, 79, 0, 34, 34, 0, 35, 35, 0, 68, 68, 29, 29, 14, 14, 52, 52, 43, 43, 80, 80, 10, 10, 78, 78, 2, 2, 9, 9, 77, 77, 15, 15, 45, 45, 39, 39, 23, 23, 63, 63, 64, 64, 16, 16, 44, 44, 28, 28, 24, 24, 19, 19, 62, 62, 70, 70, 38, 38, 3, 3, 36, 36, 5, 5, 50, 50, 76, 76, 49, 49, 60, 60, 27, 27, 42, 42, 25, 25, 53, 53, 33, 33, 47, 47, 8, 8, 51, 51, 69, 69, 20, 20, 13, 13, 74, 74, 75, 75, 48, 48, 18, 18, 66, 66, 11, 11, 71, 71, 26, 26, 59, 59, 7, 7, 65, 65, 57, 57, 4, 4, 61, 61, 54, 54, 22, 22, 12, 12])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	"""def test_feasible_file4(self):
		# Note: no feasible solution found yet
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True

	def test_feasible_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = []
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == True"""

	def test_not_feasible_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = split_a_list_at_zeros([1, 1, 0, 7, 7, 0, 5, 5, 0, 6, 6, 2, 2, 4, 4, 3, 3])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = split_a_list_at_zeros([1, 1, 0, 6, 6, 0, 12, 12, 0, 8, 8, 0, 18, 18, 0, 11, 11, 2, 2, 17, 17, 9, 9, 10, 10, 15, 15, 13, 13, 4, 4, 16, 16, 14, 14, 3, 3, 5, 5, 7, 7])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = split_a_list_at_zeros([9, 9, 0, 18, 18, 0, 31, 31, 0, 8, 8, 0, 1, 1, 0, 28, 28, 0, 26, 26, 0, 5, 5, 12, 12, 24, 24, 13, 13, 11, 11, 14, 14, 7, 7, 34, 34, 32, 32, 17, 17, 15, 15, 4, 4, 19, 19, 30, 30, 16, 16, 20, 20, 22, 22, 10, 10, 23, 23, 2, 2, 21, 21, 33, 33, 27, 27, 29, 29, 35, 35, 25, 25, 3, 3, 6, 6])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = split_a_list_at_zeros([22, 22, 0, 18, 18, 0, 21, 21, 0, 57, 57, 0, 30, 30, 0, 19, 19, 0, 26, 26, 0, 33, 33, 0, 37, 37, 0, 62, 62, 0, 8, 8, 0, 13, 13, 0, 42, 42, 0, 35, 35, 0, 31, 31, 0, 74, 74, 0, 52, 52, 0, 63, 63, 0, 34, 34, 0, 1, 1, 0, 5, 5, 27, 27, 10, 10, 79, 79, 46, 46, 49, 49, 28, 28, 6, 6, 64, 64, 36, 36, 45, 45, 75, 75, 50, 50, 65, 65, 47, 47, 55, 55, 12, 12, 59, 59, 76, 76, 20, 20, 15, 15, 43, 43, 70, 70, 7, 7, 40, 40, 54, 54, 24, 24, 39, 39, 60, 60, 41, 41, 68, 68, 80, 80, 61, 61, 38, 38, 25, 25, 48, 48, 2, 2, 73, 73, 78, 78, 23, 23, 44, 44, 9, 9, 17, 17, 67, 67, 29, 29, 56, 56, 66, 66, 4, 4, 32, 32, 77, 77, 58, 58, 53, 53, 69, 69, 71, 71, 16, 16, 72, 72, 11, 11, 14, 14, 51, 51, 3, 3])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file4(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = split_a_list_at_zeros([54, 54, 0, 31, 31, 0, 75, 75, 0, 88, 88, 0, 49, 49, 0, 90, 90, 0, 97, 97, 0, 115, 115, 0, 68, 68, 0, 119, 119, 0, 67, 67, 0, 79, 79, 0, 123, 123, 0, 112, 112, 0, 34, 34, 0, 105, 105, 0, 32, 32, 0, 51, 51, 0, 22, 22, 0, 120, 120, 0, 30, 30, 0, 83, 83, 0, 128, 128, 0, 124, 124, 0, 58, 58, 0, 10, 10, 0, 41, 41, 0, 36, 36, 0, 57, 57, 0, 107, 107, 0, 8, 8, 0, 93, 93, 0, 43, 43, 0, 89, 89, 0, 76, 76, 0, 26, 26, 0, 66, 66, 0, 81, 81, 0, 116, 116, 0, 73, 73, 0, 24, 24, 60, 60, 86, 86, 33, 33, 40, 40, 100, 100, 12, 12, 29, 29, 111, 111, 71, 71, 78, 78, 117, 117, 17, 17, 13, 13, 74, 74, 87, 87, 46, 46, 130, 130, 64, 64, 21, 21, 98, 98, 2, 2, 114, 114, 55, 55, 7, 7, 92, 92, 38, 38, 82, 82, 1, 1, 84, 84, 20, 20, 15, 15, 118, 118, 103, 103, 125, 125, 127, 127, 65, 65, 47, 47, 56, 56, 85, 85, 62, 62, 95, 95, 53, 53, 42, 42, 69, 69, 61, 61, 110, 110, 121, 121, 59, 59, 72, 72, 91, 91, 129, 129, 16, 16, 106, 106, 52, 52, 23, 23, 18, 18, 28, 28, 9, 9, 45, 45, 50, 50, 19, 19, 70, 70, 27, 27, 5, 5, 80, 80, 63, 63, 99, 99, 48, 48, 96, 96, 77, 77, 25, 25, 37, 37, 122, 122, 101, 101, 94, 94, 39, 39, 109, 109, 14, 14, 102, 102, 44, 44, 104, 104, 126, 126, 113, 113, 6, 6, 11, 11, 4, 4, 108, 108, 35, 35, 3, 3])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False

	def test_not_feasible_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = split_a_list_at_zeros([158, 158, 0, 161, 161, 0, 280, 280, 0, 298, 298, 0, 145, 145, 0, 77, 77, 0, 141, 141, 0, 5, 5, 0, 248, 248, 0, 156, 156, 0, 36, 36, 0, 187, 187, 0, 167, 167, 0, 73, 73, 0, 85, 85, 0, 21, 21, 0, 15, 15, 0, 195, 195, 0, 246, 246, 0, 103, 103, 0, 259, 259, 0, 82, 82, 0, 241, 241, 0, 186, 186, 0, 90, 90, 0, 109, 109, 0, 123, 123, 0, 104, 104, 0, 111, 111, 0, 42, 42, 0, 55, 55, 0, 152, 152, 0, 198, 198, 0, 238, 238, 0, 253, 253, 0, 78, 78, 0, 143, 143, 0, 216, 216, 0, 228, 228, 0, 59, 59, 0, 22, 22, 0, 193, 193, 0, 112, 112, 0, 212, 212, 0, 190, 190, 0, 47, 47, 0, 106, 106, 0, 220, 220, 0, 75, 75, 0, 153, 153, 0, 8, 8, 0, 164, 164, 0, 168, 168, 0, 211, 211, 0, 29, 29, 0, 57, 57, 0, 177, 177, 0, 215, 215, 0, 235, 235, 0, 244, 244, 0, 224, 224, 0, 295, 295, 0, 117, 117, 0, 196, 196, 0, 221, 221, 0, 289, 289, 0, 149, 149, 0, 283, 283, 0, 299, 299, 0, 292, 292, 0, 16, 16, 0, 191, 191, 0, 154, 154, 0, 208, 208, 0, 32, 32, 0, 213, 213, 0, 113, 113, 0, 68, 68, 0, 28, 28, 0, 148, 148, 0, 46, 46, 0, 17, 17, 0, 134, 134, 0, 122, 122, 0, 192, 192, 0, 245, 245, 0, 88, 88, 0, 48, 48, 0, 275, 275, 0, 126, 126, 0, 62, 62, 182, 182, 225, 225, 31, 31, 231, 231, 252, 252, 39, 39, 66, 66, 278, 278, 181, 181, 10, 10, 18, 18, 258, 258, 79, 79, 226, 226, 206, 206, 262, 262, 175, 175, 194, 194, 51, 51, 163, 163, 102, 102, 24, 24, 116, 116, 236, 236, 96, 96, 64, 64, 105, 105, 267, 267, 95, 95, 218, 218, 273, 273, 277, 277, 132, 132, 33, 33, 293, 293, 282, 282, 146, 146, 255, 255, 247, 247, 207, 207, 7, 7, 56, 56, 229, 229, 50, 50, 232, 232, 205, 205, 178, 178, 93, 93, 160, 160, 94, 94, 288, 288, 137, 137, 189, 189, 165, 165, 174, 174, 86, 86, 300, 300, 43, 43, 80, 80, 222, 222, 63, 63, 121, 121, 45, 45, 40, 40, 118, 118, 61, 61, 23, 23, 124, 124, 204, 204, 203, 203, 242, 242, 239, 239, 254, 254, 20, 20, 286, 286, 74, 74, 185, 185, 120, 120, 234, 234, 58, 58, 150, 150, 240, 240, 41, 41, 147, 147, 3, 3, 269, 269, 171, 171, 92, 92, 230, 230, 260, 260, 251, 251, 276, 276, 281, 281, 89, 89, 2, 2, 200, 200, 210, 210, 67, 67, 201, 201, 27, 27, 170, 170, 268, 268, 35, 35, 25, 25, 151, 151, 179, 179, 266, 266, 264, 264, 227, 227, 284, 284, 87, 87, 44, 44, 83, 83, 209, 209, 54, 54, 1, 1, 290, 290, 4, 4, 294, 294, 188, 188, 91, 91, 159, 159, 139, 139, 214, 214, 125, 125, 13, 13, 115, 115, 285, 285, 135, 135, 11, 11, 256, 256, 99, 99, 202, 202, 157, 157, 279, 279, 128, 128, 19, 19, 6, 6, 271, 271, 265, 265, 30, 30, 26, 26, 296, 296, 197, 197, 60, 60, 144, 144, 162, 162, 49, 49, 263, 263, 101, 101, 169, 169, 14, 14, 9, 9, 34, 34, 142, 142, 233, 233, 172, 172, 219, 219, 81, 81, 261, 261, 133, 133, 97, 97, 155, 155, 108, 108, 199, 199, 140, 140, 37, 37, 70, 70, 119, 119, 136, 136, 270, 270, 38, 38, 69, 69, 237, 237, 173, 173, 166, 166, 287, 287, 274, 274, 76, 76, 138, 138, 291, 291, 65, 65, 52, 52, 183, 183, 98, 98, 257, 257, 114, 114, 131, 131, 127, 127, 243, 243, 72, 72, 223, 223, 110, 110, 107, 107, 129, 129, 84, 84, 184, 184, 176, 176, 71, 71, 217, 217, 180, 180, 130, 130, 250, 250, 12, 12, 249, 249, 100, 100, 272, 272, 53, 53, 297, 297])
		feasiblity, _ = feasibility_check(sol, pytest.problem_file)

		assert feasiblity == False
		
	def test_initial_solution_file0(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[0])
		sol = initial_solution(pytest.problem_file)

		feasibility, _ = feasibility_check(sol, pytest.problem_file)

		assert feasibility == True

	def test_initial_solution_file1(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[1])
		sol = initial_solution(pytest.problem_file)

		feasibility, _ = feasibility_check(sol, pytest.problem_file)

		assert feasibility == True

	def test_initial_solution_file2(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[2])
		sol = initial_solution(pytest.problem_file)

		feasibility, _ = feasibility_check(sol, pytest.problem_file)

		assert feasibility == True

	def test_initial_solution_file3(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[3])
		sol = initial_solution(pytest.problem_file)

		feasibility, _ = feasibility_check(sol, pytest.problem_file)

		assert feasibility == True

	def test_initial_solution_file4(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[4])
		sol = initial_solution(pytest.problem_file)

		feasibility, _ = feasibility_check(sol, pytest.problem_file)

		assert feasibility == True

	def test_initial_solution_file5(self):
		pytest.problem_file = load_problem(pytest.paths_testfiles[5])
		sol = initial_solution(pytest.problem_file)

		feasibility, _ = feasibility_check(sol, pytest.problem_file)

		assert feasibility == True