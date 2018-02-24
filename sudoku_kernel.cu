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

void insertRowToSolution(int row, int* current_solution, int* quiz)
{
	for(int i = 0; i < NN; i ++)
	{
		current_solution[row*NN + i] = quiz[row*NN + i];
	}
}

resolution* createRowSolution(int row, int* current_solution, int* quiz)
{
	int *d_current_solution = copySudokuToDevice(current_solution);
	insertRowToSolution(row, current_solution, quiz);
	int *lalala = copySudokuToHost(d_current_solution);
	displayHostArray("CREATE ROW SOLUTION", lalala, NN, NN);
	displayHostArray("SOLUTION WITH INSERTED ROW", current_solution, NN, NN);

}

cudaError_t solveSudoku(int* h_sudoku_solved, int* h_sudoku_unsolved)
{
  int* resolution = new int [NN*NN];
  displayHostArray("RESOLUTION", resolution, NN, NN);

	createRowSolution(0, resolution, h_sudoku_unsolved);

	return cudaSuccess;
}

