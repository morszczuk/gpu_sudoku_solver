#include "sudoku_solver.h"

int main(int argc, char* argv[])
{
	char filename_unsolved[] = "quizzes/arr_6_unsolved.txt";
  char filename_solved[] = "quizzes/arr_1_solved.txt";
  char filename_error[] = "quizzes/arr_1_with_error.txt";
	char almost_solved[] = "quizzes/arr_1_almost_solved.txt";
	char without_two[] = "quizzes/arr_1_without_two.txt";
	char without_four[] = "quizzes/arr_1_without_four.txt";

	char silent[] = "-s";
	bool showProgress = true;
	
	int *h_sudoku_solved, *h_sudoku_unsolved, *h_sudoku_error, *h_sudoku_almost_solved, *h_without_two, *h_without_four;
	int a =5;

	if(argc > 1 )
		printf("argv[1]: %s\n", argv[1]);

	if(argc > 1 && strcmp(argv[1], "-s") == 0)
	{
		printf("Będzie mode cichy!\n");
		showProgress = false;
	}
	
	//RETRIEVING SUDOKU QUIZ

	h_sudoku_error = readSudokuArray(filename_error);
	h_sudoku_unsolved = readSudokuArray(filename_unsolved); 
  h_sudoku_solved = readSudokuArray(filename_solved);
	h_sudoku_almost_solved = readSudokuArray(almost_solved);
	h_without_two = readSudokuArray(without_two);
	h_without_four = readSudokuArray(without_four);

	//STARTING TIME MEASURMENT
	clock_t begin = clock();
	
	//SOLVING SUDOKU 
	cudaErrorHandling(solveSudoku(h_sudoku_solved, h_sudoku_unsolved, showProgress));

	//ENDING TIME MEASURMENT
	clock_t end = clock();
	printf("[FUNCTION TIME] %f ms\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);


	getchar();

	// RESETING CUDA DEVICE
	cudaErrorHandling(cudaDeviceReset());

	return 0;
}