import os

def read_file(name, readlines, **kwargs):
	with open( './input/' + name + '.txt', 'r') as file:
		for idx, line in enumerate(file.readlines()):
			kwargs = readlines(idx, line, **kwargs)
	return kwargs