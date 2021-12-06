import time
import upsidedown
import aoc.cli 
import aoc.day

def main():
	puzzles = dict()
	for function in dir(aoc.day):
		name = function.split('_')[0]
		if name == 'day':
			puzzles[function] 	= dict()
			item 				= getattr(aoc.day, function)
			if callable(item):
				start 			= time.time_ns() / 1e6
				p1, p2, f_name  = item()
				puzzles[function]["Puzzle 1"] 	= str(p1)
				puzzles[function]["Puzzle 2"] 	= str(p2)
				puzzles[function]["Name"] 		= f_name
				puzzles[function]["Time (ms)"] 	= str(time.time_ns()/1e6 - start)
	aoc.cli.make_table(**puzzles)
	aoc.cli.make_markdown(**puzzles)

if __name__ == '__main__':
	main()

