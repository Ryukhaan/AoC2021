import os
import sys
import copy
import numpy as np 
import time
from collections import defaultdict

from .cli import read_file
				
def day_1():
	def readline(idx, line, **kwargs):
		kwargs["depths"].append(int(line))
		return kwargs

	def first_solver(data):
		return len( np.argwhere(np.diff(data) > 0))

	def second_solver(data, ksize=3):
		return first_solver( np.convolve( data, np.ones(ksize), 'valid') )

	data = {"depths" : list([])}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
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
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return first_solver(data["actions"]), second_solver(data["actions"]), "Dive!"

def day_3():
	def readline(idx, line, **kwargs):
		kwargs["binaries"].append(np.array(list(line.rsplit()[0]), dtype=np.uint8))
		return kwargs

	def first_solver(data):
		numbits = len(data[0])
		most_common = np.mean(data, axis=0) >= 0.5
		gamma_rate = most_common.dot(2**np.arange(numbits)[::-1])
		epsilon_rate = (1 << numbits) - 1 - gamma_rate
		return gamma_rate * epsilon_rate

	def second_solver(data):
		numbits = len(data[0])
		oxygen_ = data
		co2_    = data
		for i in range(0, numbits):
			most_common = np.mean([binary[i] for binary in oxygen_])
			most_common = most_common >= 0.5 or 0
			oxygen_ = list(filter(lambda x: x[i]==most_common, oxygen_))
			if len(oxygen_) == 1: break
		for i in range(0, numbits):
			most_common = np.mean([binary[i] for binary in co2_])
			most_common = most_common < 0.5 or 0
			co2_ = list(filter(lambda x: x[i]==most_common, co2_))
			if len(co2_) == 1: break
		bitpack = 2**np.arange(numbits)[::-1]
		oxygen_level = oxygen_[0].dot(bitpack)
		co2_level = co2_[0].dot(bitpack)
		return oxygen_level * co2_level

	data = {"binaries" : list([])}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
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
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
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
			if x1 != x2 and y1 != y2: continue
			ymin, ymax = y1<=y2 and (y1,y2) or (y2,y1)
			xmin, xmax = x1<=x2 and (x1,x2) or (x2,x1)
			if x1 == x2:
				for j in range(ymin, ymax+1):
					accumulation[j, x1] += 1
			if y1 == y2:
				for j in range(xmin, xmax+1):
					accumulation[y1, j] += 1
		return len( np.argwhere(accumulation >= 2) ), accumulation

	def second_solver(data, accumulation):
		X, Y = np.array(data["X"]), np.array(data["Y"])
		for i in range(X.shape[0]):
			x1,y1 = X[i]
			x2,y2 = Y[i]
			dx = abs(x2-x1)+1
			if x1 < x2 and y1 < y2:
				for j in range(dx):
					accumulation[y1+j, x1+j] += 1
			elif x1 < x2 and y1 > y2:
				for j in range(dx):
					accumulation[y1-j, x1+j] += 1
			elif x1 > x2 and y1 < y2:
				for j in range(dx):
					accumulation[y1+j, x1-j] += 1
			elif x1 > x2 and y1 > y2:
				for j in range(dx):
					accumulation[y1-j, x1-j] += 1
		return len( np.argwhere(accumulation >= 2) )

	data = {"X" : list([]), "Y" : list([])}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	res, acc = first_solver(data)
	return res, second_solver(data, acc), "Hydrothermal Venture"

def day_6():
	def readline(idx, line, **kwargs):
		kwargs["lanternfish"] = np.array(line.rsplit(','))
		return kwargs

	def solver(data, i):
		M = np.array([[0,1,0,0,0,0,0,0,0],
					  [0,0,1,0,0,0,0,0,0],
					  [0,0,0,1,0,0,0,0,0],
					  [0,0,0,0,1,0,0,0,0],
					  [0,0,0,0,0,1,0,0,0],
					  [0,0,0,0,0,0,1,0,0],
					  [1,0,0,0,0,0,0,1,0],
					  [0,0,0,0,0,0,0,0,1],
					  [1,0,0,0,0,0,0,0,0]], dtype=object)
		Mn = np.linalg.matrix_power(M, i)
		lanternfishes, _ = np.histogram(data, bins=range(0,10))
		lanternfishes = np.array(lanternfishes, dtype=object)
		return sum(Mn.dot(lanternfishes.transpose()))

	data = {"lanternfish" : list([]) }
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return solver(data["lanternfish"], 80), solver(data["lanternfish"], 256), "Lanternfish"
