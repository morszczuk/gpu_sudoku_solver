#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.h"

void cudaErrorHandling(cudaError_t cudaStatus);

__global__ void checkQuizFill(int d_quiz[SUD_SIZE][SUD_SIZE], int d_fill);

__global__ void checkCorrectness(int* d_sudoku, int* d_number_presence);

cudaError_t solveSudoku(int* h_sudoku_quiz_solved, int* h_sudoku_quiz_unsolved);