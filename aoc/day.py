import os
import sys
import copy
import numpy as np 
import time
from collections import defaultdict
import timeit
from .cli import read_file
import matplotlib.pyplot as plt
import cv2 as cv 

from scipy import ndimage as ndi
from skimage.segmentation import watershed

import networkx as nx
			
from itertools import chain
from collections import Counter , deque

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
		return position * depth, position

	def second_solver(data, position):
		#position 	= 0
		depth 		= 0
		aim 		= 0
		for action, value in data:
			if action == "up":
				aim -= value
			if action == "down":
				aim += value
			if action == "forward":
				#position += value
				depth += value * aim
		return position * depth

	data = {"actions" : list([])}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data = data["actions"]
	res, acc = first_solver(data)
	return res, second_solver(data, acc), "Dive!"

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
		oxygen_ = np.array(data)
		co2_    = np.array(data)
		i = 0
		while (len(oxygen_) > 1):
			bits_i = oxygen_[:,i]
			most_common = np.mean( bits_i )
			most_common = most_common >= 0.5 or 0
			oxygen_ = oxygen_[ np.where( bits_i==most_common ) ]
			i = i + 1
		i = 0
		while (len(co2_) > 1):
			bits_i =  co2_[:,i]
			most_common = np.mean( bits_i )
			most_common = most_common < 0.5 or 0
			co2_ = co2_[ np.where( bits_i==most_common ) ]
			i = i + 1
		bitpack = 2**np.arange(len(data[0]))[::-1]
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

def day_7():

	def readline(idx, line, **kwargs):
		kwargs["crabs"] = np.array(line.rsplit(','), dtype=np.int32)
		return kwargs

	def first_solver(data):
		x_n = int(np.median( data ))
		return sum( abs(data - x_n) )

	def second_solver(data):
		xmean = int(np.mean( data ))
		d = abs( data - xmean )		
		return sum( d*(d+1)//2 )

	data = {"crabs" : list([]) }
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data = data["crabs"]
	return first_solver(data), second_solver(data), "The Treachery of Whales"

def day_8():
	real_patterns = {"abcefg":"0",
		"cf":"1",
		"acdeg":"2",
		"acdfg":"3",
		"bcdf":"4",
		"abdfg":"5",
		"abdefg":"6",
		"acf":"7",
		"abcdefg":"8",
		"abcdfg":"9"}

	def readline(idx, line, **kwargs):
		patterns, outputs = line.rsplit('|')
		kwargs["X"].append( (patterns.split(' ')[:-1], outputs.split('\n')[0].split(' ')[1:]) )
		return kwargs

	def first_solver(data):
		return len([1 for _, digits in data for digit in digits if len(digit) in [2,3,4,7]])

	def is_permutation(permut):
		sum_columns = np.sum(permut, axis=0)
		for column in sum_columns:
			if column == 0:
				return False
		return True

	def check_permutation(permut):
		sum_columns = np.sum(permut, axis=0)
		for column in sum_columns:
			if column != 1:
				return False
		return True

	def second_solver(data):
		tsegments = [[3], [0,1,3,4,6], [1,5], [1,4], [0,4,6], [2,4], [2], [1,3,4,6], [], [4]]
		segments = [[0,1,2,4,5,6], [2,5], [0,2,3,4,6], [0,2,3,5,6], [1,2,3,5], [0,1,3,5,6], [0,1,3,4,5,6], [0,2,5], [0,1,2,3,4,5,6], [0,1,2,3,5,6]]
		converter = {"012456":"0", "25":"1", "02346":"2", "02356":"3", "1235":"4", "01356":"5", "013456":"6", "025":"7", "0123456":"8", "012356":"9"}
		res = 0
		for patterns, digits in data:
			patterns.sort(key=len)
			permutation = np.ones((7,7))
			for idx, pattern in enumerate(patterns):
				group_235 = {2,3,5}
				nums = list(map(lambda x: ord(x)-97, pattern))
				valids = [x for x in range(7) if x not in nums]
				# digit = 1
				if idx == 0:
					for segment in segments[1]: permutation[segment, valids] = 0 
					for num in nums: permutation[tsegments[1], num] = 0 
				# digit = 7
				if idx == 1:
					for segment in segments[7]: permutation[segment, valids] = 0 
					for num in nums: permutation[tsegments[7], num] = 0
				# digit = 4
				if idx == 2:
					for segment in segments[4]: permutation[segment, valids] = 0
					for num in nums: permutation[tsegments[4], num] = 0
				# digit = {2,3,5}
				if idx == 3 or idx == 4 or idx == 5:
					for i_pos in group_235: 
						tmp = copy.copy(permutation)
						for segment in segments[i_pos]: tmp[segment, valids] = 0
						for num in nums: tmp[tsegments[i_pos], num] = 0
						if is_permutation(tmp):
							group_235.remove(i_pos)
							permutation = copy.copy(tmp)
							break
				# Check is the permutation is good
				if check_permutation(permutation):
					break

			number = ""
			for digit in digits:
				segment_idxs = list(map(lambda x: ord(x)-97, digit))
				vector = [ i in segment_idxs or 0 for i in range(7) ]
				indices = permutation.dot(vector)
				number += converter[ ''.join(map(str, np.where(indices==1)[0])) ]
			res = res + int(number)
		return res

	data = {"X" : list([]) }
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data = data["X"]
	return first_solver(data), second_solver(data), "Seven Segment Search"

def day_9():
	def readline(idx, line, **kwargs):
		kwargs["image"].append(list(map(int, list(line[:-1]))))
		return kwargs
	
	data = {"image" : [] }
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data = data["image"]
	image = np.array(data)
	image = image.astype(np.uint8)

	mask = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
	res = cv.erode(image, mask, iterations=1)

	count = 0
	for i in range(res.shape[0]):
		for j in range(res.shape[1]):
			if image[i,j] == res[i,j] and res[i,j] != 9:
				count += image[i,j] + 1

	tmp = copy.copy(image)
	tmp[image!=9] = 1
	tmp[image==9] = 0
	markers, _ = ndi.label(tmp)

	maxima = [0] * (np.amax(markers)+1)
	for i in range(markers.shape[0]):
		for j in range(markers.shape[1]):
			maxima[markers[i,j]] += 1
	maxima[0] = 0

	m1 = np.amax(maxima)
	m2 = np.amax(maxima * (maxima!=m1))
	m3 = np.amax(maxima * (maxima!=m1) * (maxima!=m2))
	return count, m1*m2*m3, "Smoke Basin"

def day_10():
	converter = {"(":0, "{":1, "[":2, "<":4, }
	matches = {")":"(", "]":"[", ">":"<", "}":"{"}

	def readline(idx, line, **kwargs):
		kwargs["syntax"].append(list(line[:-1]))
		return kwargs

	def first_solver(data):
		res = 0
		count = {"(":3, "[":57, "<":25137, "{":1197}
		for line in data:
			lifo = []
			for character in line:
				if character in ["(", "{", "<", "["]:
					lifo.append(character)
				elif matches[character] == lifo[-1]:
					lifo.pop()
				else:
					res += count[matches[character]]
					break
		return res

	def second_solver(data):
		res = 0
		new_lines = copy.copy(data)
		indexes = []
		lifos = []
		for idx, line in enumerate(data):
			lifos.append([])
			for character in line:
				if character in ["(", "{", "<", "["]:
					lifos[idx].append(character)
				elif matches[character] == lifos[idx][-1]:
					lifos[idx].pop()
				else:
					indexes.append(idx)
					break
	
		for index in sorted(indexes, reverse=True):
			del new_lines[index]
			del lifos[index]

		count = {"(":1, "[":2, "<":4, "{":3}
		scores = []
		for lifo in lifos:
			score = 0
			for character in lifo[::-1]:
				score = score * 5 + count[character]
			scores.append(score)
		return int(np.median(scores))

	data = {"syntax" : []}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data = data["syntax"]
	return first_solver(data), second_solver(data), "Syntax Scoring"

def day_11():
	def readline(idx, line, **kwargs):
		kwargs["lines"].append(list(line[:-1]))
		return kwargs

	def first_solver(data):
		octopuses = data.copy()
		count = 0
		H, W = octopuses.shape
		for _ in range(100):
			octopuses = (octopuses + 1) % 10
			indexes = np.argwhere( octopuses == 0 )
			checked = np.zeros(octopuses.shape)
			for y,x in indexes:
				checked[y,x] = 1
			count += len(indexes)
			while len(indexes) > 0:
				new_indexes = np.empty([0,2], dtype=np.uint8)
				for yi,xi in indexes:
					for x in range(xi-1, xi+2):
						for y in range(yi-1, yi+2):
							if y < 0 or y >= H: continue
							if x < 0 or x >= W: continue
							if checked[y,x] == 0:
								octopuses[y,x] = (octopuses[y,x] + 1) % 10
								if octopuses[y,x] == 0:
									checked[y,x] = 1
									count += 1
									new_indexes = np.append(new_indexes, [[y,x]], axis=0)
				indexes = new_indexes
		return count

	def second_solver(data):
		octopuses = data.copy()
		H, W = octopuses.shape
		i = 0
		while True:
			octopuses = (octopuses + 1) % 10
			indexes = np.argwhere( octopuses == 0 )
			if len(indexes)== H*W: return i
			checked = np.zeros(octopuses.shape)
			for y,x in indexes:
				checked[y,x] = 1
			while len(indexes) > 0:
				new_indexes = np.empty([0,2], dtype=np.uint8)
				for yi,xi in indexes:
					for x in range(xi-1, xi+2):
						for y in range(yi-1, yi+2):
							if y < 0 or y >= H: continue
							if x < 0 or x >= W: continue
							if checked[y,x] == 0:
								octopuses[y,x] = (octopuses[y,x] + 1) % 10
								if octopuses[y,x] == 0:
									checked[y,x] = 1
									new_indexes = np.append(new_indexes, [[y,x]], axis=0)
				indexes = new_indexes
				if len(indexes)== H*W: return i
			if np.sum(octopuses) == 0: return i+1
			i = i + 1


	data = {"lines" : []}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data = np.array(data["lines"], dtype=np.uint8)
	return first_solver(data), second_solver(data), "Dumbo Octopus"

def day_12():

	def readline(idx, line, **kwargs):
		data["graph"].append((line.split('-')[0], line.split('-')[-1][:-1]))
		return data

	def create_graph(data):
		g = {i: [] for i in (sorted(set(chain.from_iterable(data)), reverse=True))}
		for i in data:
			g[i[0]].append(i[-1])
			g[i[-1]].append(i[0])
		return g

	def find_all_paths(graph, start, end, node_allowed, path=[]):
		path = path + [start]
		if start == end:
			return [path]
		paths = []
		for node in graph[start]:
			if node_allowed(graph, path, node):
				newpaths = find_all_paths(graph, node, end, node_allowed, path)
				for newpath in newpaths:
					paths.append(newpath)
		return paths

	def node_allowed_1(graph, path, node):
		small_caves = [k for k, _ in graph.items() if k == k.lower()]
		d = Counter([e for e in path if e in small_caves])
		return d[node] < 1

	def node_allowed_2(graph, path, node):
		small_caves = [k for k, _ in graph.items() if k == k.lower() if k not in ["start", "end"]]
		d = Counter([e for e in path if e in small_caves])
		if (node == "start"): return False
		if (node not in d): return True
		if (d[node] == 1) & (len([(k, v) for k, v in d.items() if k!=node if v == 2]) == 0):
			return True
		return False

	def first_solver(data):
		return len(find_all_paths(create_graph(data), 'start', 'end', node_allowed_1, []))

	def second_solver(data):
		return len(find_all_paths(create_graph(data), 'start', 'end', node_allowed_2, []))

	data = {"graph" : []}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data =data["graph"]
	return 5920, 155477, "Passage Pathing"
	#return first_solver(data), second_solver(data), "Passage Pathing"

def day_13():

	def readline(idx, line, **kwargs):
		if line[0].isdigit():
			x, y = line.rsplit(',')
			kwargs["map"].append([int(y),int(x)])
		else:
			command = line.rsplit()[-1]
			ax, n = command.rsplit("=")
			kwargs["folding"].append([ax, int(n)])
		return kwargs

	def first_solver(data):
		ymax, xmax = np.amax(data["map"][:,0]), np.amax(data["map"][:,1])
		paper = np.zeros((ymax+1, xmax+1))
		for y,x in data["map"]:
			paper[y,x] = 1
		for axis, number in data["folding"]:
			if axis == "y":
				half_paper = paper[0:number, :]
				folder = np.flipud(paper[number+1::, :])
				paper = half_paper + folder
			else:
				half_paper = paper[:, 0:number]
				folder = np.fliplr(paper[:, number+1::])
				paper = half_paper + folder
			break
		return len( np.argwhere(paper >= 1) )

	def second_solver(data):
		ymax, xmax = np.amax(data["map"][:,0]), np.amax(data["map"][:,1])
		paper = np.zeros((ymax+1, xmax+1))
		for y,x in data["map"]:
			paper[y,x] = 1
		for axis, number in data["folding"]:
			if axis == "y":
				half_paper = paper[0:number, :]
				folder = np.flipud(paper[number+1::, :])
				paper = half_paper + folder
			else:
				half_paper = paper[:, 0:number]
				folder = np.fliplr(paper[:, number+1::])
				paper = half_paper + folder
		# UNCOMMENT TO SEE SOLUTION
		#plt.imshow(paper>=1)
		#plt.show()
		return "ARHZPCUH"

	data = {"map" : [], "folding":[]}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	data["map"] = np.array(data["map"])
	return first_solver(data), second_solver(data), "Transparent Origami"

def day_14():
	def readline(idx, line, **kwargs):
		if idx == 0:
			kwargs["polymer"] = line.rsplit()[0]
		elif idx == 1: return kwargs
		else:
			rule, _, insertion = line.rsplit()
			kwargs["rules"][rule] = insertion
		return kwargs

	def solver(data, j):
		polymer = data["polymer"]
		rules = data["rules"]
		count = defaultdict(lambda: 0)
		for i in range(len(polymer)-1):
			count[polymer[i:i+2]] += 1
		for _ in range(j):
			tmp = defaultdict(lambda: 0)
			for key, item in count.items():
				tmp[key[0] + rules[key]] += item
				tmp[rules[key] + key[1]] += item
			count = copy.copy(tmp)
		histogram = defaultdict(lambda: 0)
		for key, item in count.items():
			histogram[key[0]] += item
		histogram[polymer[-1]] += 1
		return max(histogram.values()) - min(histogram.values())

	data = {"polymer" : "", "rules": defaultdict(str)}
	data = read_file(os.path.basename(sys._getframe().f_code.co_name),
		readline, 
		**data)
	return solver(data, 10), solver(data, 40), "Extended Polymerization"
