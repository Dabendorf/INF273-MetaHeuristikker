from Utils import *

prob = load_problem("../Data/Call_7_Vehicle_3.txt")

sol = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]

print(prob.keys())
"""for k, v in prob.items():
	print(f"{k} {v}")"""

print(prob["vehicle_calls"])
exit(0)

feasiblity, c = feasibility_check(sol, prob)

Cost = cost_function(sol, prob)

print(feasiblity)
print(c)
print(Cost)
