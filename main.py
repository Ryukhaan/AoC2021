import os
import time

from collections import defaultdict

import input_reader as io
import day

import upsidedown

def make_markdown(**kwargs):
	header_str = "|  {: {}{}} | {: {}{}} | {: {}{}} | {: {}{}} |"
	row_string = "| ʕノ•ᴥ•ʔノ {: {}{}} | {: {}{}} | {: {}{}} | {: {}{}} |"
	result_filename = './README.md'

	space_col = 20
	text = []
	text.append( header_str.format("Day", '^', space_col,
									"Puzzle 1", '^', space_col,
									"Puzzle 2", "^", space_col,
									"Time (ms)", "^", space_col))
	text.append("|:-----:|:----------:|:--------:|-----------:|")
	# text.append( header_str.format("", "^", space_col,
	# 	"Puzzle 1", "^", space_col,
	# 	"Puzzle 2", "^", space_col,
	# 	"Time (ms)", "^", space_col))

	for idx, (key, value) in enumerate(kwargs.items()):
		day_number = key.split('_')[1]
		day_str = upsidedown.transform("Day " + day_number)
		# text.append( d_name_str.format(day_str, '<', space_col, 
		# 							   '', '>', space_col,
		# 							   '', '>', space_col,
		# 							   '', '>', space_col) )
		text.append( row_string.format(upsidedown.transform(value["Name"]), '>', space_col, 
									value["Puzzle 1"], '^', space_col,
									value["Puzzle 2"], '^', space_col,
									value["Time (ms)"], '^', space_col) )
	with open( result_filename, 'w') as file:
		file.write( "\n".join(text) )

def make_table(**kwargs):
	header_str = "| {: {}{}} | {: {}{}} |"
	column_str = "| {: {}{}} | {: {}{}} | {: {}{}} | {: {}{}} |"
	d_name_str = "| ʕノ•ᴥ•ʔノ {: {}{}} | {: {}{}} | {: {}{}} | {: {}{}} |"

	text = []
	maximum_len = 107
	espace_column = (maximum_len - 7)//4
	n_tiret = "─"* espace_column

	v_line = "|" + "·"* espace_column + "├" + n_tiret + ("┼" + n_tiret)*2 + "┤"
	# Header 
	text.append( "┌" + n_tiret + "┬" + (n_tiret + "─")*2 + n_tiret + "┐" )
	text.append( header_str.format("Day", '^', espace_column-2, 
								   "Results", '^', 3*espace_column) )
	text.append( "├" + n_tiret + "┼" + n_tiret + ("┬" + n_tiret)*2 + "┤" )
	text.append( column_str.format("", '^', espace_column-2, 
								   "Puzzle 1", '^', espace_column-2,
								   "Puzzle 2", '^', espace_column-2,
								   "Time (ms)", '^', espace_column-2) )
	text.append( v_line )
	
	# Body
	num_key = len(kwargs.keys())
	for idx, (key, value) in enumerate(kwargs.items()):
		day_number = key.split('_')[1]
		day_str = upsidedown.transform("Day " + day_number)
		text.append( d_name_str.format(day_str, '<', espace_column-12, 
									   '', '>', espace_column-2,
									   '', '>', espace_column-2,
									   '', '>', espace_column-2) )
		text.append( column_str.format(value["Name"], '>', espace_column-2, 
									value["Puzzle 1"], '^', espace_column-2,
									value["Puzzle 2"], '^', espace_column-2,
									value["Time (ms)"], '^', espace_column-2) )
		if idx < num_key-1:
			text.append( v_line )

	# Footer
	text.append("└" + n_tiret + ("┴" + n_tiret)*3 + "┘")
	text = '\n'.join(text)
	print( text )


def main():
	puzzles = dict()
	for function in dir(day):
		name = function.split('_')[0]
		if name == 'day':
			puzzles[function] 	= dict()
			item 				= getattr(day, function)
			if callable(item):
				start 			= time.time_ns() / 1e6
				p1, p2, f_name  = item()
				puzzles[function]["Puzzle 1"] 	= str(p1)
				puzzles[function]["Puzzle 2"] 	= str(p2)
				puzzles[function]["Name"] 		= f_name
				puzzles[function]["Time (ms)"] 	= str(time.time_ns()/1e6 - start)
	make_table(**puzzles)
	make_markdown(**puzzles)

if __name__ == '__main__':
	main()

