#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.h"
// #include "helpers.h"

struct resolution {
  int n;
  int** resolutions;
};

void displayHostArray(char* title, int* array, int N, int M);
void cudaErrorHandling(cudaError_t cudaStatus);
cudaError_t solveSudoku(int* h_sudoku_solved, int* h_sudoku_unsolved);