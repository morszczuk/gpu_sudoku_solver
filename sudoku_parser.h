#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>

#include "constants.h"

int* readSudokuArray(char* filename);

//printing Array in sudoku-style.
void printArray(int* array, int N, int M);