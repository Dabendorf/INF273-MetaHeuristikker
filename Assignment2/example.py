from Utils import *
# from pdp_utils.Utils import load_problem
# from UtilsOld import load_problem, feasibility_check, cost_function

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	prob = load_problem("../Data/Call_7_Vehicle_3.txt")
	logger.info("Problem reading finished")
	
	sol = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]

	print(prob.keys())
	"""for k, v in prob.items():
		print(f"{k} {v}")"""

	print(prob["vehicle_calls"])

	feasiblity, c = feasibility_check(sol, prob)
	logger.info("Feasibility Check finished")

	Cost = cost_function(sol, prob)
	logger.info("Cost function finished")

	print(feasiblity)
	print(c)
	print(Cost)

if __name__ == "__main__":
	main()