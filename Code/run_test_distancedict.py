from Utils import *

import logging
import time

def main():
	logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.disabled = False

	test_files = ["../Data/Call_7_Vehicle_3.txt", "../Data/Call_18_Vehicle_5.txt", "../Data/Call_35_Vehicle_7.txt", "../Data/Call_80_Vehicle_20.txt", "../Data/Call_130_Vehicle_40.txt", "../Data/Call_300_Vehicle_90.txt"]
	#test_files = ["../Data/Call_7_Vehicle_3.txt"]

	start = time.time()
	for test_f in test_files:
		file = test_f
		prob = load_problem(file)
		logger.info(f"Problem reading finished, file {file}")

		dist_diff_dict = generate_dist_dict(prob)
		"""for k,v in dist_diff_dict.items():
			print(f"{k}, {v}")"""
		print(len(dist_diff_dict))
		print(f"Time: {time.time()-start}")
		start = time.time()

if __name__ == "__main__":
	main()