import os
import sys
import copy
import numpy as np 
import time
from collections import defaultdict

import input_reader as io
				
def day_1():
	def readline(idx, line, **kwargs):
		kwargs["depths"].append(int(line))
		return kwargs

	def first_solver(data):
		return sum( [(data[i+1] - data[i]) > 0 for i in range(len(data)-1)] )

	def second_solver(data, ksize=3):
		mean_k = [ sum(data[i:i+ksize]) for i in range(len(data)-ksize+1) ]
		return first_solver(mean_k)

	data = {"depths" : list([])}
	data = io.read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return first_solver(data["depths"]), second_solver(data["depths"]), "Sonar Sweep"

def day_2():
	command = {"forward" : 1, "up" : -1, "down" : 1}
	def readline(idx, line, **kwargs):
		action, x = line.split(' ')
		kwargs["actions"].append((action, int(x)))
		return kwargs

	def first_solver(data):
		position = sum([value for action, value in data if action == "forward"])
		depth 	 = sum([command[action] * value for action, value in data if action in ["up","down"]])
		return position * depth

	def second_solver(data):
		position 	= 0
		depth 		= 0
		aim 		= 0
		for action, value in data:
			if action == "up":
				aim -= value
			if action == "down":
				aim += value
			if action == "forward":
				position += value
				depth += value * aim
		return position * depth

	data = {"actions" : list([])}
	data = io.read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return first_solver(data["actions"]), second_solver(data["actions"]), "Dive!"

def day_3():
	def readline(idx, line, **kwargs):
		kwargs["binaries"].append(line.rsplit()[0])
		return kwargs

	def first_solver(data):
		numbits = len(data[0])
		gamma_rate = [0] * numbits
		for i in range(0, numbits):
			most_common = np.mean([int(binary[i]) for binary in data])
			gamma_rate[i] = int(most_common >= 0.5 or 0)
		gamma_rate = ''.join(map(str, gamma_rate))
		gamma_rate = int(gamma_rate, 2)
		epsilon_rate = (1 << numbits) - 1 - gamma_rate
		return gamma_rate * epsilon_rate

	def second_solver(data):
		max_num = len(data[0])
		oxygen_ = copy.copy(data)
		co2_    = copy.copy(data)
		for i in range(0, max_num):
			most_common = np.mean([int(binary[i]) for binary in oxygen_])
			most_common = most_common >= 0.5 or 0
			oxygen_ = list(filter(lambda x: int(x[i])==most_common, oxygen_))
			if len(oxygen_) == 1: break
		for i in range(0, max_num):
			most_common = np.mean([int(binary[i]) for binary in co2_])
			most_common = most_common < 0.5 or 0
			co2_ = set(filter(lambda x: int(x[i])==most_common, co2_))
			if len(co2_) == 1: break
		return int(oxygen_.pop(),2) * int(co2_.pop(),2)

	data = {"binaries" : list([])}
	data = io.read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return first_solver(data["binaries"]), second_solver(data["binaries"]), "Binary Diagnostic"

def day_4():
	def readline(idx, line, **kwargs):
		if idx == 1: return kwargs
		if idx == 0:
			kwargs["numbers"] = np.array(line.rsplit(','), dtype=np.uint8)
			return kwargs
		if (idx-1)%6 == 0:
			kwargs["bingos"].append( (np.array(kwargs["bingo"]), np.zeros((5,5))) )
			kwargs["bingo"] = []
			return kwargs
		else:
			kwargs["bingo"].append(np.array(line.rsplit(), dtype=np.uint8))
			return kwargs
		return kwargs

	def as_win(bingo):
		sum_columns = np.sum(bingo, axis=0)
		for column in sum_columns:
			if column == 5: return True
		sum_rows 	= np.sum(bingo, axis=1)
		for row in sum_rows:
			if row == 5: return True
		return False

	def check_winners(bingos):
		for bingo in bingos:
			if as_win(bingo[1]): return True, bingo
		return False, None

	def check_number(bingo, marked, number):
		array = np.argwhere(bingo==number)
		if not array.size == 0:
			marked[array[0,0],array[0,1]] = 1
		return (bingo, marked)

	def check_numbers(bingos, numbers):
		for number in numbers:
			for idx, bingo in enumerate(bingos):
				bingos[idx] = check_number(bingo[0], bingo[1], number)
			as_winner, winner = check_winners(bingos)
			if as_winner: return winner, number
		return None

	def check_last_numbers(bingos, numbers, winners):
		n = len(bingos)
		last_win = None
		last_num = None
		for number in numbers:
			for idx, bingo in enumerate(bingos):
				if winners[idx] == 0:
					bingos[idx] = check_number(bingo[0], bingo[1], number)
					if as_win(bingo[1]): 
						winners[idx] = 1
						last_win = bingo
						last_num = number
		return last_win, last_num

	def first_solver(data):
		winner, num = check_numbers(data["bingos"], data["numbers"])
		array = np.argwhere( winner[1]==0 )
		total = sum([winner[0][row[0],row[1]] for row in array])
		return total * num

	def second_solver(data):		
		winner, num = check_last_numbers(data["bingos"], 
			data["numbers"], [0]*len(data["bingos"]))
		array = np.argwhere( winner[1]==0 )
		total = sum([winner[0][row[0],row[1]] for row in array])
		return total * num

	data = {"numbers" : list([]), "bingos" : list([]), "bingo" : list([])}
	data = io.read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return first_solver(data), second_solver(data), "Giant Squid"

def day_5():
	def readline(idx, line, **kwargs):
		X, Y = line.rsplit('->')
		kwargs["X"].append( np.array(X.split(','), dtype=np.int16) )
		kwargs["Y"].append( np.array(Y.split(','), dtype=np.int16) )
		return kwargs

	def first_solver(data):
		X, Y = np.array(data["X"]), np.array(data["Y"])
		xmax, ymax = np.amax(X), np.amax(Y)
		accumulation = np.zeros((ymax+1,xmax+1))
		for i in range(X.shape[0]):
			x1,y1 = X[i]
			x2,y2 = Y[i]
			ymin = y1 if y1 <= y2 else y2
			ymax = y1 if y1 >= y2 else y2
			xmin = x1 if x1 <= x2 else x2
			xmax = x1 if x1 >= x2 else x2
			if x1 == x2:
				for j in range(ymin, ymax+1):
					accumulation[j, x1] += 1
			if y1 == y2:
				for j in range(xmin, xmax+1):
					accumulation[y1, j] += 1
		return len( np.argwhere(accumulation >= 2) )

	def second_solver(data):
		X, Y = np.array(data["X"]), np.array(data["Y"])
		xmax, ymax = np.amax(X), np.amax(Y)
		accumulation = np.zeros((ymax+1,xmax+1))
		for i in range(X.shape[0]):
			x1,y1 = X[i]
			x2,y2 = Y[i]
			ymin = y1 if y1 <= y2 else y2
			ymax = y1 if y1 >= y2 else y2
			xmin = x1 if x1 <= x2 else x2
			xmax = x1 if x1 >= x2 else x2
			if x1 == x2:
				for j in range(ymin, ymax+1):
					accumulation[j, x1] += 1
			if y1 == y2:
				for j in range(xmin, xmax+1):
					accumulation[y1, j] += 1
			if x1 < x2 and y1 < y2:
				for j in range(x2-x1+1):
					accumulation[y1+j, x1+j] += 1
			elif x1 < x2 and y1 > y2:
				for j in range(x2-x1+1):
					accumulation[y1-j, x1+j] += 1
			elif x1 > x2 and y1 < y2:
				for j in range(x1-x2+1):
					accumulation[y1+j, x1-j] += 1
			elif x1 > x2 and y1 > y2:
				for j in range(x1-x2+1):
					accumulation[y1-j, x1-j] += 1
		return len( np.argwhere(accumulation >= 2) )

	data = {"X" : list([]), "Y" : list([])}
	data = io.read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return first_solver(data), second_solver(data), "Hydrothermal Venture"
