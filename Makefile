all: sudoku_solver
    
sudoku_solver:
	nvcc kernel.cu sudoku_parser.cpp -o sudoku_solver

clean:
	rm -f *.o