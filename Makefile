all: sudoku_solver
    
sudoku_solver:
	nvcc sudoku_kernel.cu sudoku_solver.cpp sudoku_parser.cpp -arch=sm_30 -o sudoku_solver

clean:
	rm -f *.o