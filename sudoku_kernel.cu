#include "sudoku_kernel.h"

void cudaErrorHandling(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error on CUDA %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
}

void displayHostArray(char* title, int* array, int N, int M)
{
  printf("---------%s-----------\n", title);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
			printf("%d |", array[i*N + j]);
		
		printf("\n");

		for (int j = 0; j < N; j++)
			printf("- |");
		
		printf("\n");
	}
  printf("------------------------------\n");
}

int* copySudokuToDevice(int* h_sudoku)
{
	int* d_sudoku;
	
	cudaErrorHandling(cudaMalloc((void **)&d_sudoku, NN * NN * sizeof(int)));

	cudaErrorHandling(cudaMemcpy(d_sudoku, h_sudoku, NN * NN * sizeof(int), cudaMemcpyHostToDevice));

	return d_sudoku;
}

int* copySudokuToHost(int* d_sudoku)
{
	int* h_sudoku = new int[NN*NN];

	cudaErrorHandling(cudaMemcpy(h_sudoku, d_sudoku, NN * NN * sizeof(int), cudaMemcpyDeviceToHost));

	return h_sudoku;
}

int* copySudoku(int* sudoku)
{
	int* sudokuCopy = new int[NN*NN];
	for(int i = 0; i < NN*NN; i++)
	{
		sudokuCopy[i] = sudoku[i];
	}
	return sudokuCopy;
}

int* insertRowToSolution(int row, int* current_solution, int* quiz)
{
	int* solution_copy = copySudoku(current_solution);
	for(int i = 0; i < NN; i ++)
	{
		solution_copy[row*NN + i] = quiz[row*NN + i];
	}
	return solution_copy;
}

resolution* createRowSolution(int row, int* _current_solution, int* quiz)
{
	int* current_solution, *d_current_solution;
	resolution* created_resolution = new resolution();

	current_solution = insertRowToSolution(row, _current_solution, quiz);
	
	d_current_solution = copySudokuToDevice(current_solution);

	if(row == 8)
	{
		created_resolution -> n = 1;
		created_resolution -> resolutions = current_solution;
		return created_resolution;
	} else
	{
		return createRowSolution(row + 1, current_solution, quiz);
	}
}

cudaError_t solveSudoku(int* h_sudoku_solved, int* h_sudoku_unsolved)
{
  int* empty_resolution = new int [NN*NN];
	resolution* final_resolution;
  displayHostArray("RESOLUTION", empty_resolution, NN, NN);

	final_resolution = createRowSolution(0, empty_resolution, h_sudoku_unsolved);
	printf("Wynikow: %d", final_resolution -> n);

	return cudaSuccess;
}

