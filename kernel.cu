#include "kernel.h"

void cudaErrorHandling(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error on CUDA %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
}

__global__ void checkQuizFill(int d_quiz[SUD_SIZE][SUD_SIZE], int d_fill)
{
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;

	//] = d_quiz[idx][idy] > 0 ? 1 : 0;
}

__global__ void checkCorrectness(int* d_sudoku, int* d_number_presence)
{
	extern __shared__ int number_presence[];
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;
	int index_1, index_2, index_3;
	int k = 81;

	number_presence[idx * 9 + idy] = 0;
	number_presence[k + idx * 9 + idy] = 0;
	number_presence[(2*k) + (idx * 9 + idy)] = 0;

	index_1 = idx * 9 + d_sudoku[idx/SUD_SIZE + idy] - 1;
	index_2 = k + idy * 9 + d_sudoku[idx/SUD_SIZE + idy] - 1;
	index_3 = (2 * k) + ((idx / 3) * 27) + ((idy / 3) * 9) + d_sudoku[idx/SUD_SIZE + idy] - 1;

	printf("[idx: %d, idy: %d | val: %d | %d, %d, %d]\n", idx, idy, d_sudoku[idx/SUD_SIZE + idy], index_1, index_2 - k , index_3 - (2*k));
}

cudaError_t solveSudoku(int* h_sudoku_quiz)
{
	int *d_sudoku_quiz, *d_quiz_fill, *d_number_presence;
	int sharedMemorySize;
	cudaErrorHandling(cudaMalloc((void **)&d_sudoku_quiz, SUD_SIZE * SUD_SIZE * sizeof(int)));
	cudaErrorHandling(cudaMalloc((void **)&d_quiz_fill, SUD_SIZE * SUD_SIZE * sizeof(int)));

	cudaErrorHandling(cudaMemcpy(d_sudoku_quiz, h_sudoku_quiz, SUD_SIZE * SUD_SIZE * sizeof(int), cudaMemcpyHostToDevice));

	cudaErrorHandling(cudaMalloc((void **)&d_number_presence, 243 * sizeof(int)));

	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);
	sharedMemorySize = 243 * sizeof(int);
	checkCorrectness <<<dimGrid, dimBlock, sharedMemorySize>>> (d_sudoku_quiz, d_number_presence);
	cudaErrorHandling(cudaDeviceSynchronize());
	//int h_sudoku_quiz[SUD_SIZE][SUD_SIZE];

	//for(int i = 0; i < SUD_SIZE; i++)
	//	for (int j = 0; j < SUD_SIZE; j++)
	//		h_sudoku_quiz[i][j] = _sudoku_quiz[i][j];
}