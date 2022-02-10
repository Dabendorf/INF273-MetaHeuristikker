from Utils import *

import logging

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	test_files = ["../Data/Call_7_Vehicle_3.txt", "../Data/Call_18_Vehicle_5.txt", "../Data/Call_35_Vehicle_7.txt", "../Data/Call_80_Vehicle_20.txt", "../Data/Call_130_Vehicle_40.txt", "../Data/Call_300_Vehicle_90.txt"]
	file = test_files[0]
	prob = load_problem(file)
	logger.info(f"Problem reading finished, file {file}")
	
	feasiblity, rand_sol, cost, counter = blind_random_search(prob)
	logger.info(f"Generate solution with blind search")
	logger.info(f"Solution: {rand_sol}")
	logger.info(f"Costs: {cost}")

if __name__ == "__main__":
	main()