import os
import sys
import copy
import numpy as np 
import time
from collections import defaultdict

from ..cli import read_file
from .common_day import CommonDay
		
class Day(CommonDay):
	"""docstring for Day"""
	def __init__(self):
		super(Day, self).__init__()
		self.name = "Sonar Sweep"
		self.initialize()

	def readline(self, idx, line, **kwargs):
		kwargs["depths"].append(int(line))
		return kwargs

	def initialize(self):
		data = {"depths" : list([])}
		data = read_file("day_1",
			self.readline,
			**data)
		self.data = data["depths"] 

	def first_solver(self):
		return len( np.argwhere(np.diff(self.data) > 0))

	def second_solver(self, ksize=3):
		return len(np.argwhere(np.diff(np.convolve(self.data, np.ones(ksize), 'valid'))>0))