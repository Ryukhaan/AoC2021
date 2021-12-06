import os
import time

from collections import defaultdict

import input_reader as io
import day

import upsidedown

def make_table(**kwargs):
	header_str = "| {: {}{}} | {: {}{}} |"
	column_str = "| {: {}{}} | {: {}{}} | {: {}{}} |"
	d_name_str = "| ʕノ•ᴥ•ʔノ {: {}{}} | {: {}{}} | {: {}{}} |"

	text = []
	maximum_len = 88
	espace_column = (maximum_len - 7)//3
	result_filename = './results.txt'
	n_tiret = "─"* espace_column

	# Header 
	text.append( "┌" + n_tiret + "┬" + n_tiret + "─" + n_tiret + "┐" )
	text.append( header_str.format("Day", '^', espace_column-2, 
								   "Results", '^', 2*espace_column-1) )
	text.append( "├" + n_tiret + "┼" + n_tiret + "┬" + n_tiret + "┤" )
	text.append( column_str.format("", '^', espace_column-2, 
								   "Puzzle 1", '^', espace_column-2,
								   "Puzzle 2", '^', espace_column-2) )
	text.append( "|" + "·"* espace_column + ("├" + n_tiret)*2 + "┤" )
	
	# Body
	num_key = len(kwargs.keys())
	for idx, (key, value) in enumerate(kwargs.items()):
		day_number = key.split('_')[1]
		day_str = upsidedown.transform("Day " + day_number)
		text.append( d_name_str.format(day_str, '<', espace_column-12, 
									   '', '>', espace_column-2,
									   '', '>', espace_column-2) )
		text.append( column_str.format(value["Name"], '>', espace_column-2, 
									value["Puzzle 1"], '^', espace_column-2,
									value["Puzzle 2"], '^', espace_column-2) )
		if idx < num_key-1:
			text.append( "|" + "·"* espace_column + ("├" + n_tiret)*2 + "┤" )

	# Footer
	text.append("└" + n_tiret + "┴" + n_tiret + "┴" + n_tiret + "┘")
	text = '\n'.join(text)
	#with open( result_filename, 'w') as file:
	#	file.write(text)
	print( text )


def main():
	puzzles = dict()
	for function in dir(day):
		name = function.split('_')[0]
		if name == 'day':
			puzzles[function] 	= dict()
			item 				= getattr(day, function)
			if callable(item):
				start 			= time.time_ns()
				p1, p2, f_name  = item()
				puzzles[function]["Puzzle 1"] 	= str(p1)
				puzzles[function]["Puzzle 2"] 	= str(p2)
				puzzles[function]["Name"] 		= f_name
				puzzles[function]["Time (ns)"] 	= str(time.time_ns() - start)
	make_table(**puzzles)

if __name__ == '__main__':
	main()

