from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	filename = "../Data/Call_300_Vehicle_90.txt"
	prob = load_problem(filename)
	logger.info("Problem reading finished")
	
	feasiblity, rand_sol, cost, counter = blind_random_search(prob)
	print(filename)
	print(feasiblity)
	print(rand_sol)
	print(cost)
	print(counter)

if __name__ == "__main__":
	main()