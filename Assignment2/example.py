from Utils import *
# from pdp_utils.Utils import load_problem
# from UtilsOld import load_problem, feasibility_check, cost_function

prob = load_problem("../Data/Call_7_Vehicle_3.txt")

sol = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]

print(prob.keys())
"""for k, v in prob.items():
	print(f"{k} {v}")"""

print(prob["vehicle_calls"])

feasiblity, c = feasibility_check(sol, prob)

Cost = cost_function(sol, prob)

print(feasiblity)
print(c)
print(Cost)
