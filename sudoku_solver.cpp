#include "sudoku_solver.h"

int main(int argc, char* argv[])
{
	char filename_unsolved[] = "quizzes/arr_1_unsolved.txt";
  char filename_solved[] = "quizzes/arr_1_solved.txt";
	int * h_sudoku_quiz;
	int a =5;
	
	//RETRIEVING SUDOKU QUIZ
  if(argc > 1)
	  h_sudoku_quiz = readSudokuArray(filename_unsolved);
  else
    h_sudoku_quiz = readSudokuArray(filename_solved);
	
  printArray(h_sudoku_quiz, SUD_SIZE, SUD_SIZE);

	//STARTING TIME MEASURMENT
	clock_t begin = clock();
	
	//SOLVING SUDOKU 
	cudaErrorHandling(solveSudoku(h_sudoku_quiz));

	//ENDING TIME MEASURMENT
	clock_t end = clock();
	printf("[FUNCTION TIME] %f ms\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);


	getchar();

	// RESETING CUDA DEVICE
	cudaErrorHandling(cudaDeviceReset());

	return 0;
}