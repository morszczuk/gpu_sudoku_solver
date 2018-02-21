#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include "constants.h"
#include "sudoku_parser.h"

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
	printf("[idx: %d | idx: %d ]\n", idx, idy);
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

int main()
{
	char filename[] = "quizzes/arr_1_solved.txt";
	int * h_sudoku_quiz;
	int a =5;
	
	//RETRIEVING SUDOKU QUIZ
	h_sudoku_quiz = readSudokuArray(filename);
	printArray(h_sudoku_quiz, SUD_SIZE, SUD_SIZE);

	//STARTING TIME MEASURMENT
	clock_t begin = clock();
	
	//SOLVING SUDOKU 
	cudaErrorHandling(solveSudoku(h_sudoku_quiz));
	// if (cudaStatus != cudaSuccess) {
	// 	printf("fds");
	// 	fprintf(stderr, "solveSudoku failed!");
	// 	return 1;
	// }

	//ENDING TIME MEASURMENT
	clock_t end = clock();
	printf("[FUNCTION TIME] %f ms\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);


	getchar();

	// RESETING CUDA DEVICE
	cudaErrorHandling(cudaStatus = cudaDeviceReset());
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaDeviceReset failed!");
	// 	return 1;
	// }

	return 0;
}