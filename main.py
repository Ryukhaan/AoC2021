import os
import time

from collections import defaultdict

import input_reader as io
import day

def make_table(**kwargs):
	template_str = "| {: {}{}} | {: {}{}} |"
	result_str = "| {: {}{}} | ʕ•ᴥ•ʔノ{: {}{}} |"
	# Header 
	maximum_len = 51
	print("="*maximum_len)
	header_dict = ["Puzzles", "Results"]
	dk = (maximum_len - 7)//2
	print(template_str.format("Puzzles", '<', dk, "Results", '>', dk))
	print("="*maximum_len)
	
	# Body
	for key, value in kwargs.items():
		num = key.split('_')[1]
		print(template_str.format("Day " + num, '<', dk, '', '>', dk))
		for day, result in value.items():
			print(result_str.format(day, '^', dk, result, '>', dk-7))

	# Footer
	print("="*maximum_len)

def main():
	puzzles = dict()
	for function in dir(day):
		name = function.split('_')[0]
		if name == 'day':
			puzzles[function] = dict()
			item = getattr(day, function)
			if callable(item):
				start = time.time_ns()
				p1, p2 = item()
				puzzles[function]["Puzzle 1"] = str(p1)
				puzzles[function]["Puzzle 2"] = str(p2)
				puzzles[function]["Time"] = str(time.time_ns() - start)
	print(puzzles)
	make_table(**puzzles)

if __name__ == '__main__':
	main()

