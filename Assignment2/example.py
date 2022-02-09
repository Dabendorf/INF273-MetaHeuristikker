from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	prob = load_problem("../Data/Call_80_Vehicle_20.txt")
	logger.info("Problem reading finished")
	
	counter = 0
	for i in range(10000):
		sol = random_solution(prob)
		feasiblity, c = feasibility_check(sol, prob)
		if feasiblity:
			counter += 1
		"""print(i)
		feasiblity, c = feasibility_check(i, prob)
		print(c)
		if feasiblity:
			print(i)
			cost = cost_function(i, prob)
			print(f"Cost: {cost}")
			counter += 1"""
	print(counter)

	# feasiblity, c = feasibility_check(sol, prob)
	# logger.info("Feasibility Check finished")

	# cost = cost_function(sol, prob)
	# logger.info("Cost function finished")

	# print(feasiblity)
	# print(c)
	#print(cost)

if __name__ == "__main__":
	main()