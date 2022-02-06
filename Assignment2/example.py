from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	prob = load_problem("../Data/Call_7_Vehicle_3.txt")
	logger.info("Problem reading finished")
	
	sol = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]

	feasiblity, c = feasibility_check(sol, prob)
	logger.info("Feasibility Check finished")

	# cost = cost_function(sol, prob)
	# logger.info("Cost function finished")

	print(feasiblity)
	print(c)
	#print(cost)

if __name__ == "__main__":
	main()