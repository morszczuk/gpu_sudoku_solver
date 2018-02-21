#include "sudoku_parser.h"

int* readSudokuArray(char* filename)
{
	int* h_sudoku = new int[SUD_SIZE*SUD_SIZE];

	//printf("SUDOKU FILENAME: %s\n", filename);
	std::ifstream sudoku_file(filename);

	int a0, a1, a2, a3, a4, a5, a6, a7, a8;
	int i = 0;

	while (sudoku_file >> a0 >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8)
	{
		h_sudoku[i + 0] = a0;
		h_sudoku[i + 1] = a1;
		h_sudoku[i + 2] = a2;
		h_sudoku[i + 3] = a3;
		h_sudoku[i + 4] = a4;
		h_sudoku[i + 5] = a5;
		h_sudoku[i + 6] = a6;
		h_sudoku[i + 7] = a7;
		h_sudoku[i + 8] = a8;
		i++;
	}

	return h_sudoku;
}

//printing Array in sudoku-style.
void printArray(int* array, int N, int M)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
			printf("%d |", array[i/N + j]);
		
		printf("\n");

		for (int j = 0; j < N; j++)
			printf("- |");
		
		printf("\n");
	}
}