from unittest import TestCase

from Utils import split_a_list_at_zeros
import pytest
	
class HelperfunctionValidation(TestCase):
	def test_list_splitting_at_zero1(self):
		a = [0, 2, 2, 0, 1, 5, 5, 3, 1, 3, 0, 7, 4, 6, 7, 4, 6]

		assert split_a_list_at_zeros(a) == [[], [2, 2], [1, 5, 5, 3, 1, 3], [7, 4, 6, 7, 4, 6]]

	def test_list_splitting_at_zero2(self):
		b = [0, 7, 7, 0, 0, 2, 6, 4, 1, 6, 3, 4, 5, 5, 2, 1, 3]

		assert split_a_list_at_zeros(b) == [[], [7, 7], [], [2, 6, 4, 1, 6, 3, 4, 5, 5, 2, 1, 3]]

	def test_list_splitting_at_zero3(self):
		c = [0, 0, 6, 6, 0, 3, 3, 5, 2, 1, 4, 2, 7, 7, 1, 5, 4]

		assert split_a_list_at_zeros(c) == [[], [], [6, 6], [3, 3, 5, 2, 1, 4, 2, 7, 7, 1, 5, 4]]

	def test_list_splitting_at_zero4(self):
		d = [7, 7, 0, 2, 6, 4, 1, 6, 3, 4, 5, 5, 2, 1, 3, 0, 0]

		assert split_a_list_at_zeros(d) == [[7, 7], [2, 6, 4, 1, 6, 3, 4, 5, 5, 2, 1, 3], []]

	def test_list_splitting_at_zero5(self):
		e = [7, 7, 0, 2, 6, 4, 1, 6, 3, 4, 0, 5, 5, 2, 1, 3, 0]

		assert split_a_list_at_zeros(e) == [[7, 7], [2, 6, 4, 1, 6, 3, 4], [5, 5, 2, 1, 3]]
