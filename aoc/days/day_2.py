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
		self.name = "Dive!"
		self.initialize()

	def readline(self, idx, line, **kwargs):
		action, x = line.split(' ')
		kwargs["actions"].append((action, int(x)))
		return kwargs

	def initialize(self):
		data = {"actions" : list([])}
		data = read_file("day_2",
			self.readline, 
			**data)
		self.data = data["actions"]
		self.command = {"forward" : 1, "up" : -1, "down" : 1}

	def first_solver(self):
		position = sum([value for action, value in self.data if action == "forward"])
		depth 	 = sum([self.command[action] * value for action, value in self.data if action in ["up","down"]])
		return position * depth

	def second_solver(self):
		position 	= 0
		depth 		= 0
		aim 		= 0
		for action, value in self.data:
			if action == "up":
				aim -= value
			if action == "down":
				aim += value
			if action == "forward":
				position += value
				depth += value * aim
		return position * depth