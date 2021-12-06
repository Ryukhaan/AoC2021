import upsidedown
import os

def read_file(name, readlines, **kwargs):
	filepath = os.path.abspath(os.path.dirname(__file__))
	input_path = os.path.join(filepath, '../input/' + name + '.txt')
	with open( input_path, 'r' ) as file:
		for idx, line in enumerate(file.readlines()):
			kwargs = readlines(idx, line, **kwargs)
	return kwargs

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
	text.append("|:-----|:----------:|:--------:|-----------:|")
	for idx, (key, value) in enumerate(kwargs.items()):
		day_number = key.split('_')[1]
		day_str = upsidedown.transform("Day " + day_number)
		text.append( row_string.format(upsidedown.transform(value["Name"]), '<', space_col, 
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

