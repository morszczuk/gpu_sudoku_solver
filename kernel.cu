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

//reading sudoku quiz from a file
// int* readSudokuArray(char* filename)
// {
// 	int* h_sudoku = new int[SUD_SIZE*SUD_SIZE];

// 	//printf("SUDOKU FILENAME: %s\n", filename);
// 	std::ifstream sudoku_file(filename);

// 	int a0, a1, a2, a3, a4, a5, a6, a7, a8;
// 	int i = 0;

// 	while (sudoku_file >> a0 >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8)
// 	{
// 		h_sudoku[i + 0] = a0;
// 		h_sudoku[i + 1] = a1;
// 		h_sudoku[i + 2] = a2;
// 		h_sudoku[i + 3] = a3;
// 		h_sudoku[i + 4] = a4;
// 		h_sudoku[i + 5] = a5;
// 		h_sudoku[i + 6] = a6;
// 		h_sudoku[i + 7] = a7;
// 		h_sudoku[i + 8] = a8;
// 		i++;
// 	}

// 	return h_sudoku;
// }

// //printing Array in sudoku-style.
// void printArray(int* array, int N, int M)
// {
// 	for (int i = 0; i < N; i++)
// 	{
// 		for (int j = 0; j < M; j++)
// 			printf("%d |", array[i/N + j]);
		
// 		printf("\n");

// 		for (int j = 0; j < N; j++)
// 			printf("- |");
		
// 		printf("\n");
// 	}
// }

__global__ void checkQuizFill(int d_quiz[SUD_SIZE][SUD_SIZE], int d_fill)
{
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;

	//] = d_quiz[idx][idy] > 0 ? 1 : 0;
}

__global__ void checkCorrectness(int* d_sudoku, int* d_number_presence)
{
	fprintf(stdout, "madafaka3\n");
	extern __shared__ int number_presence[];
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;
	printf("idx: %d", idx);
	printf("idy: %d", idy);
}

cudaError_t solveSudoku(int* h_sudoku_quiz)
{
	int *d_sudoku_quiz, *d_quiz_fill, *d_number_presence;
	int sharedMemorySize;
	int f;
	printf("madafaka\n");
	cudaMalloc((void **)&d_sudoku_quiz, SUD_SIZE * SUD_SIZE * sizeof(int));
	cudaMalloc((void **)&d_quiz_fill, SUD_SIZE * SUD_SIZE * sizeof(int));

	cudaMemcpy(d_sudoku_quiz, h_sudoku_quiz, SUD_SIZE * SUD_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_number_presence, 243 * sizeof(int));
	printf("madafaka2\n");

	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);
	sharedMemorySize = 243 * sizeof(int);
	fprintf(stdout, "madafaka3\n");
	checkCorrectness <<<dimGrid, dimBlock, sharedMemorySize>>> (d_sudoku_quiz, d_number_presence);
	cudaDeviceSynchronize();
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
	cudaError_t cudaStatus = solveSudoku(h_sudoku_quiz);
	if (cudaStatus != cudaSuccess) {
		printf("fds");
		fprintf(stderr, "solveSudoku failed!");
		return 1;
	}

	//ENDING TIME MEASURMENT
	clock_t end = clock();
	printf("[FUNCTION TIME] %f ms\n", (double)(end - begin) / CLOCKS_PER_SEC * 1000);


	getchar();

	// RESETING CUDA DEVICE
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}