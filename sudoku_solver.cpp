#include "sudoku_solver.h"

int main(int argc, char* argv[])
{
	char filename_unsolved[] = "quizzes/arr_1_unsolved.txt";
  char filename_solved[] = "quizzes/arr_1_solved.txt";
  char filename_error[] = "quizzes/arr_1_with_error.txt";
	char almost_solved[] = "quizzes/arr_1_almost_solved.txt";
	int *h_sudoku_solved, *h_sudoku_unsolved, *h_sudoku_error, *h_sudoku_almost_solved;
	int a =5;
	
	//RETRIEVING SUDOKU QUIZ

	h_sudoku_error = readSudokuArray(filename_error);
    
	h_sudoku_unsolved = readSudokuArray(filename_unsolved);
  
  h_sudoku_solved = readSudokuArray(filename_solved);
	h_sudoku_almost_solved = readSudokuArray(almost_solved);
	
  printArray(h_sudoku_unsolved, SUD_SIZE, SUD_SIZE);
	printArray(h_sudoku_almost_solved, SUD_SIZE, SUD_SIZE);

	//STARTING TIME MEASURMENT
	clock_t begin = clock();
	
	//SOLVING SUDOKU 
	cudaErrorHandling(solveSudoku(h_sudoku_solved, h_sudoku_unsolved));

	//ENDING TIME MEASURMENT
	clock_t end = clock();
	printf("[FUNCTION TIME] %f ms\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);


	getchar();

	// RESETING CUDA DEVICE
	cudaErrorHandling(cudaDeviceReset());

	return 0;
}