#include "sudoku_kernel.h"

bool solution_found = false;

void cudaErrorHandling(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error on CUDA %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
}

int* newArrayWithZero(int N, int M)
{
	int* empty_array = new int [N*M];

	for(int i = 0; i < N*M; i++)
		empty_array[i] = 0;

	return empty_array;
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

int* copyArrayToDevice(int* h_array, int size)
{
	int* d_array;
	
	cudaErrorHandling(cudaMalloc((void **)&d_array, size * sizeof(int)));

	cudaErrorHandling(cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice));

	return d_array;
}

int* copySudokuToDevice(int* h_sudoku)
{
	return copyArrayToDevice(h_sudoku, NN * NN);
}

int* copySudokuToHost(int* d_sudoku)
{
	int* h_sudoku = new int[NN*NN];

	cudaErrorHandling(cudaMemcpy(h_sudoku, d_sudoku, NN * NN * sizeof(int), cudaMemcpyDeviceToHost));

	return h_sudoku;
}

int* duplicateSudoku(int* sudoku)
{
	int* sudokuCopy = new int[NN*NN];
	for(int i = 0; i < NN*NN; i++)
	{
		sudokuCopy[i] = sudoku[i];
	}
	return sudokuCopy;
}

__global__ void __sumNumberPresenceInRow(int* d_number_presence, int row)
{
	int idx = threadIdx.x;

	if( idx % NN == 8)
		return;

	for (int i = 1 ; i <= NN / 2; i *= 2)
	{
		if (idx % (2 * i) == 0) {
			// printf("BEFORE [Thread %d]: %d + %d\n", idx + row*NN, d_number_presence[idx + row*NN], d_number_presence[idx + i + row*NN]);
			d_number_presence[idx + row*NN] += d_number_presence[idx + i + row*NN];
			// printf("AFTER [Thread %d]: %d\n", idx + row*NN, d_number_presence[idx+ row*NN]);
		}
		else
		{
			// printf("[Thread %d] returning\n", idx + row*NN);
			return;
		}
		__syncthreads();
	}

	if(idx == 0)
		d_number_presence[idx + row*NN] += d_number_presence[8 + row*NN];
}

__global__ void __fillNumberPresenceArray(int* d_sudoku, int* d_number_presence)
{
	extern __shared__ int number_presence[];
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;
	// int index_1, index_2, index_3;
	int k = SUD_SIZE*SUD_SIZE;

	number_presence[idx * SUD_SIZE + idy] = 0;
	number_presence[k + idx * SUD_SIZE + idy] = 0;
	number_presence[(2*k) + (idx * SUD_SIZE + idy)] = 0;

	// index_1 = idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_2 = k + idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_3 = (2 * k) + ((idx / 3) * 27) + ((idy / 3) * SUD_SIZE) + d_sudoku[idx*SUD_SIZE + idy] - 1;

	// printf("[idx: %d, idy: %d | val: %d | %d, %d, %d]\n", idx, idy, d_sudoku[idx*SUD_SIZE + idy], index_1, index_2 - k , index_3 - (2*k));

	__syncthreads();

	if (d_sudoku[idx*SUD_SIZE + idy])
	{
		number_presence[idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs about number data[idx][idy] - 1 presence in row idx
		number_presence[k + (idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1)] = 1; //informs about number data[idx][idy] - 1 presence in column idy
		number_presence[(2 * k) + ((idx / 3) * 27) + ((idy / 3) * 9) + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs, that number which is in data[idx][idy] - 1 is present in proper 'quarter'
	}

	__syncthreads();

	d_number_presence[idx * SUD_SIZE + idy] = number_presence[idx * 9 + idy];
	d_number_presence[k + idx * SUD_SIZE + idy] = number_presence[k + idx * 9 + idy];
	d_number_presence[(2 * k) + (idx * SUD_SIZE + idy)] = number_presence[(2 * k) + (idx * 9 + idy)];
	
	__syncthreads();
}


__global__ void __fillElementPresenceInRowsArray(int* d_sudoku, int* d_number_presence_in_rows)
{
	extern __shared__ int number_presence[];
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;
	// int index_1, index_2, index_3;

	number_presence[idx * SUD_SIZE + idy] = 0;
	// number_presence[k + idx * SUD_SIZE + idy] = 0;
	// number_presence[(2*k) + (idx * SUD_SIZE + idy)] = 0;

	// index_1 = idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_2 = k + idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_3 = (2 * k) + ((idx / 3) * 27) + ((idy / 3) * SUD_SIZE) + d_sudoku[idx*SUD_SIZE + idy] - 1;

	// printf("[idx: %d, idy: %d | val: %d | %d, %d, %d]\n", idx, idy, d_sudoku[idx*SUD_SIZE + idy], index_1, index_2 - k , index_3 - (2*k));

	__syncthreads();

	if (d_sudoku[idx * NN + idy])
	{
		number_presence[idx * NN + idy] = 1; //informs about number data[idx][idy] - 1 presence in row idx
		// number_presence[k + (idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1)] = 1; //informs about number data[idx][idy] - 1 presence in column idy
		// number_presence[(2 * k) + ((idx / 3) * 27) + ((idy / 3) * 9) + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs, that number which is in data[idx][idy] - 1 is present in proper 'quarter'
	}

	__syncthreads();

	d_number_presence_in_rows[idx * NN + idy] = number_presence[idx * NN + idy];
	// d_number_presence[k + idx * SUD_SIZE + idy] = number_presence[k + idx * 9 + idy];
	// d_number_presence[(2 * k) + (idx * SUD_SIZE + idy)] = number_presence[(2 * k) + (idx * 9 + idy)];
	
	__syncthreads();
}


__global__ void __fillNumberPresenceInRowsArray(int* d_sudoku, int* d_number_presence_in_rows)
{
	extern __shared__ int number_presence[];
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;
	// int index_1, index_2, index_3;

	number_presence[idx * SUD_SIZE + idy] = 0;
	// number_presence[k + idx * SUD_SIZE + idy] = 0;
	// number_presence[(2*k) + (idx * SUD_SIZE + idy)] = 0;

	// index_1 = idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_2 = k + idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_3 = (2 * k) + ((idx / 3) * 27) + ((idy / 3) * SUD_SIZE) + d_sudoku[idx*SUD_SIZE + idy] - 1;

	// printf("[idx: %d, idy: %d | val: %d | %d, %d, %d]\n", idx, idy, d_sudoku[idx*SUD_SIZE + idy], index_1, index_2 - k , index_3 - (2*k));

	__syncthreads();

	if (d_sudoku[idx*SUD_SIZE + idy])
	{
		number_presence[idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs about number data[idx][idy] - 1 presence in row idx
		// number_presence[k + (idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1)] = 1; //informs about number data[idx][idy] - 1 presence in column idy
		// number_presence[(2 * k) + ((idx / 3) * 27) + ((idy / 3) * 9) + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs, that number which is in data[idx][idy] - 1 is present in proper 'quarter'
	}

	__syncthreads();

	d_number_presence_in_rows[idx * SUD_SIZE + idy] = number_presence[idx * 9 + idy];
	// d_number_presence[k + idx * SUD_SIZE + idy] = number_presence[k + idx * 9 + idy];
	// d_number_presence[(2 * k) + (idx * SUD_SIZE + idy)] = number_presence[(2 * k) + (idx * 9 + idy)];
	
	__syncthreads();
}

__device__ void __sumNumberPresence(int* d_number_presence_in_col, int size)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	// printf("[IDX: %d]\n", idx);
	__syncthreads();
	if(threadIdx.x < 64)
	{
		for (int i = 1; i <= 64 / 2; i *= 2)
		{
			if (threadIdx.x % (2 * i) == 0) {
				// printf("BEFORE [Thread %d]: %d\n", idx, d_number_presence_in_col[idx]);
				d_number_presence_in_col[idx] += d_number_presence_in_col[idx + i];
				// printf("AFTER [Thread %d]: %d\n", idx, d_number_presence_in_col[idx]);
			}
			else
			{
				// printf("[Thread %d] returning\n", idx);
				return;
			}
			__syncthreads();
		}
	} else if(threadIdx.x < 80)
	{
		int id = threadIdx.x - 64;
		for(int i = 1; i <= 8; i *= 2)
		{
			if (id % (2 * i) == 0) {
				// printf("BEFORE [Thread %d]: %d\n", idx, d_number_presence_in_col[idx]);
				d_number_presence_in_col[idx] += d_number_presence_in_col[idx + i];
				// printf("AFTER [Thread %d]: %d\n", idx, d_number_presence_in_col[idx]);
			}
			else
			{
				// printf("[Thread %d] returning\n", idx);
				return;
			}
			__syncthreads();
		}

	}

	__syncthreads();

	if(threadIdx.x == 0)
	{
		d_number_presence_in_col[idx] += d_number_presence_in_col[idx + 64];
		d_number_presence_in_col[idx] += d_number_presence_in_col[idx + 80];
		// printf("FINALNY WYNIK w komórce 0: %d\n", d_number_presence_in_col[idx]);
	}

}

__global__ void __checkAlternativeSolutionsCorrectness(int* d_alternative_solutions_one_array, bool* d_alternative_solutions_correctness, int* d_number_presence_in_col, int row)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	// int row = threadIdx.x % NN;
	int col = threadIdx.x - ((threadIdx.x / NN)*NN);
	int blockStart = blockDim.x*blockIdx.x;

	// printf("Moje IDX: %d", idx);

	d_number_presence_in_col[idx] = 0;

	if(threadIdx.x == 0)
		d_alternative_solutions_correctness[blockIdx.x] = false;

	__syncthreads();

	// printf("IDX: %d | WARTOSC: %d | COL: %d | INDEKS DO WSTAIWENIA: %d\n", idx, d_alternative_solutions_one_array[idx], col, blockStart + (col * NN) + d_alternative_solutions_one_array[idx] - 1);
	if(d_alternative_solutions_one_array[idx] > 0)
	{
		// printf("AKTUALNA WARTOSC: %d\n", d_number_presence_in_col[blockStart + (col * NN) + d_alternative_solutions_one_array[idx] - 1]);
		d_number_presence_in_col[blockStart + (col * NN) + d_alternative_solutions_one_array[idx] - 1] += 1; //informs about number data[idx][idy] - 1 presence in column idy
	}
	//number_presence[k + (idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1)] = 1; //informs about number data[idx][idy] - 1 presence in column idy
	//d_number_presence_in_row[rowStart + d_alternative_solutions_one_array[idx] - 1] += 1;

	__syncthreads();

	__sumNumberPresence(d_number_presence_in_col, 81);
	__syncthreads();

	if(threadIdx.x == 0)
		if(d_number_presence_in_col[blockIdx.x*blockDim.x] != (row+1)*NN)
			d_alternative_solutions_correctness[blockIdx.x] = true;

}

__global__ void __sumNumberPresenceArray(int* d_number_presence, int size)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	// printf("[IDX: %d]\n", idx);
	__syncthreads();

	for (int i = 1; i <= size / 2; i *= 2)
	{
		if (idx % (2 * i) == 0) {
			// printf("BEFORE [Thread %d]: %d\n", idx, d_number_presence[idx]);
			d_number_presence[idx] += d_number_presence[idx + i];
			// printf("AFTER [Thread %d]: %d\n", idx, d_number_presence[idx]);
		}
		else
		{
			// printf("[Thread %d] returning\n", idx);
			return;
		}
		__syncthreads();
	}

	if(idx == 0)
		d_number_presence[idx] += d_number_presence[idx + 128];
}

__global__ void __fillFullNumberPresenceArray(int* d_sudoku, int* d_number_presence)
{
	extern __shared__ int number_presence[];
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;
	// int index_1, index_2, index_3;
	int k = SUD_SIZE*SUD_SIZE;

	number_presence[idx * SUD_SIZE + idy] = 0;
	number_presence[k + idx * SUD_SIZE + idy] = 0;
	number_presence[(2*k) + (idx * SUD_SIZE + idy)] = 0;

	// index_1 = idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_2 = k + idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1;
	// index_3 = (2 * k) + ((idx / 3) * 27) + ((idy / 3) * SUD_SIZE) + d_sudoku[idx*SUD_SIZE + idy] - 1;

	// printf("[idx: %d, idy: %d | val: %d | %d, %d, %d]\n", idx, idy, d_sudoku[idx*SUD_SIZE + idy], index_1, index_2 - k , index_3 - (2*k));

	__syncthreads();

	if (d_sudoku[idx*SUD_SIZE + idy])
	{
		number_presence[idx * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs about number data[idx][idy] - 1 presence in row idx
		number_presence[k + (idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1)] = 1; //informs about number data[idx][idy] - 1 presence in column idy
		number_presence[(2 * k) + ((idx / 3) * 27) + ((idy / 3) * 9) + d_sudoku[idx*SUD_SIZE + idy] - 1] = 1; //informs, that number which is in data[idx][idy] - 1 is present in proper 'quarter'
	}

	__syncthreads();

	d_number_presence[idx * SUD_SIZE + idy] = number_presence[idx * 9 + idy];
	d_number_presence[k + idx * SUD_SIZE + idy] = number_presence[k + idx * 9 + idy];
	d_number_presence[(2 * k) + (idx * SUD_SIZE + idy)] = number_presence[(2 * k) + (idx * 9 + idy)];
	
	__syncthreads();
}

int* fillNumberPresenceInRowsArray(int* d_sudoku) 
{
	int* d_number_presence_in_rows;
	int sharedMemorySize = NN*NN * sizeof(int);
	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);

	cudaErrorHandling(cudaMalloc((void **)&d_number_presence_in_rows, NN*NN * sizeof(int)));

	__fillNumberPresenceInRowsArray <<<dimGrid, dimBlock, sharedMemorySize>>> (d_sudoku, d_number_presence_in_rows);
	cudaErrorHandling(cudaDeviceSynchronize());

	//displayNumberPresenceArray(d_number_presence);

	return d_number_presence_in_rows;
}

int* fillElementPresenceInRowsArray(int* d_sudoku) 
{
	int* d_element_presence_in_rows;
	int sharedMemorySize = NN*NN * sizeof(int);
	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);

	cudaErrorHandling(cudaMalloc((void **)&d_element_presence_in_rows, NN*NN * sizeof(int)));

	__fillElementPresenceInRowsArray <<<dimGrid, dimBlock, sharedMemorySize>>> (d_sudoku, d_element_presence_in_rows);
	cudaErrorHandling(cudaDeviceSynchronize());

	//displayNumberPresenceArray(d_number_presence);

	return d_element_presence_in_rows;
}


int sumNumberPresenceInRow(int* d_number_presence, int row)
{
	int* summing_result = new int[NN*NN];
	dim3 dimBlock2 = dim3(9, 1, 1);
	dim3 dimGrid2 = dim3(1);

	__sumNumberPresenceInRow <<<dimGrid2, dimBlock2>>> (d_number_presence, row);
	cudaErrorHandling(cudaDeviceSynchronize());

	cudaErrorHandling(cudaMemcpy(summing_result, d_number_presence, NN*NN * sizeof(int), cudaMemcpyDeviceToHost));

	return summing_result[row*NN];
}

int* defineNumbersToInsert(int numbers_to_insert_amount, int* h_number_presence, int row)
{	
	int* numbers_to_insert = new int[numbers_to_insert_amount];

	int i = 0;
	int j = row*NN;

	while(i < numbers_to_insert_amount)
	{
		if(h_number_presence[j] == 0)
		{
			// printf("O, dodaję element!!! Liczba do wstawienia: %d\n", (j % NN) + 1);
			numbers_to_insert[i] = (j % NN) + 1;
			i++;
		}
		j++;
	}

	return numbers_to_insert;
}


int* definePositionsToInsert(int numbers_to_insert_amount, int* h_element_presence, int row)
{	
	int* positions_to_insert = new int[numbers_to_insert_amount];

	int i = 0;
	int j = row*NN;

	while(i < numbers_to_insert_amount)
	{
		if(h_element_presence[j] == 0)
		{
			// printf("Pozycja do wstawienia: %d\n", j % NN);
			positions_to_insert[i] = j % NN;
			i++;
		}
		j++;
	}

	return positions_to_insert;
}

int countEmptyElemsInRow(int row, int* d_number_presence)
{
	// int* d_number_presence = fillNumberPresenceInRowsArray(d_current_solution);
	// int* h_number_presence = copySudokuToHost(d_number_presence);
	int filled_elements = sumNumberPresenceInRow(d_number_presence, row);

	// int* numbersToInsert = defineNumbersToInsert(NN - filled_elements, h_number_presence, row);

	// int* d_element_presence = fillElementPresenceInRowsArray(d_current_solution);
	// int* h_element_presence = copySudokuToHost(d_element_presence);
	// int* positions_to_insert = definePositionsToInsert(NN - filled_elements, h_element_presence, row);

	// printf("LICZBA ELEMENTÓW WYPEŁNIONYCH w rzędzie %d: %d\n", row + 1, filled_elements);

	return NN - filled_elements;
}

int factorial(int n)
{
	if( n <= 1) return n;
	return n*factorial(n-1);
}

int** createPermutations(int empty_elems_in_row)
{
	int permutations[empty_elems_in_row];
	int n_factorial = factorial(empty_elems_in_row);
	int** result = new int*[n_factorial];

	for(int i = 0; i < n_factorial; i++)
	{
		result[i] = new int[empty_elems_in_row];
	}

	for(int i = 0; i < empty_elems_in_row; i ++)
	{
		permutations[i] = i;
	}

	// printf("PERMUTUJEMY\n");

	int i = 0;
	do
	{
		for(int j = 0; j < empty_elems_in_row; j++)
		{
			result[i][j] = permutations[j];
			// printf("%d | ", permutations[j]);
		}
		// printf("\n");
		i++;
	} while (std::next_permutation(permutations, permutations + empty_elems_in_row));

	return result;
}

int* insertPossibleSolutionToRow(int* h_current_solution, int num_of_elements_to_insert, int* positions_to_insert, int* numbers_to_insert, int* permutation, int row)
{
	int* possibleSolution = duplicateSudoku(h_current_solution);
	for(int i = 0; i < num_of_elements_to_insert; i++)
	{
		possibleSolution[NN*row + positions_to_insert[i]] = numbers_to_insert[permutation[i]];
	}
	// printf("ALTERNATYWNE ROZWIAZANIE STWORZONE!!!!\n");
	// displayHostArray("ALTERNATYWNE ROZWWIAZANIE", possibleSolution, NN, NN);

	return possibleSolution;
}

int** createAlternativeSolutions(int* h_current_solution, int num_of_elements_to_insert, int* positions_to_insert, int* numbers_to_insert, int** rowPermutations, int row)
{
	int n_factorial = factorial(num_of_elements_to_insert);
	int** alternative_solutions = new int*[n_factorial];

	for(int i = 0; i < n_factorial; i++)
	{
		alternative_solutions[i] = insertPossibleSolutionToRow(h_current_solution, num_of_elements_to_insert, positions_to_insert, numbers_to_insert, rowPermutations[i], row);
	}

	return alternative_solutions;
}

int* combineSolutionsIntoOneArray(int n_factorial, int** alternative_solutions)
{
	int* solutions_array = new int[n_factorial*NN*NN];

	for(int i = 0; i < n_factorial; i ++)
		for(int j = 0; j < NN*NN; j++)
			solutions_array[j + i*NN*NN] = alternative_solutions[i][j];

	return solutions_array;
}


bool* checkAlternativeSolutionsCorrectness(int row, int n_factorial, int* alternative_solutions_one_array)
{
	int* d_alternative_solutions_one_array = copyArrayToDevice(alternative_solutions_one_array, n_factorial * NN * NN);
	int* d_number_presence_in_row;
	bool* d_alternative_solutions_correctness, *h_alternative_solutions_correctness;
	h_alternative_solutions_correctness = new bool[n_factorial];
	
	
	cudaErrorHandling(cudaMalloc((void **)&d_alternative_solutions_correctness, n_factorial * sizeof(bool)));
	cudaErrorHandling(cudaMalloc((void **)&d_number_presence_in_row, n_factorial * NN * NN * sizeof(int)));

	dim3 dimBlock = dim3(81, 1, 1);
	dim3 dimGrid = dim3(n_factorial);

	__checkAlternativeSolutionsCorrectness <<<dimGrid, dimBlock>>>(d_alternative_solutions_one_array, d_alternative_solutions_correctness, d_number_presence_in_row, row);
	cudaErrorHandling(cudaDeviceSynchronize());

	cudaErrorHandling(cudaMemcpy(h_alternative_solutions_correctness, d_alternative_solutions_correctness, n_factorial * sizeof(bool), cudaMemcpyDeviceToHost));
	
	cudaFree(d_alternative_solutions_correctness);
	cudaFree(d_alternative_solutions_one_array);

	return h_alternative_solutions_correctness;
}

resolution* chooseCorrectSolutions(int n_factorial, int** alternative_solutions, bool* alternative_solutions_correctness)
{
	resolution* only_correct_solutions = new resolution();
	int number_of_correct = 0;

	for(int i = 0; i < n_factorial; i++)
		if(!alternative_solutions_correctness[i])
			number_of_correct++;

	int** correct_solutions_arr = new int*[number_of_correct];
	
	int k = 0;
	for(int i = 0; i < n_factorial; i++)
		if(!alternative_solutions_correctness[i])
		{
			correct_solutions_arr[k] = alternative_solutions[i];
			k++;
		}
	
	only_correct_solutions -> n = number_of_correct;
	only_correct_solutions -> resolutions = correct_solutions_arr;

	return only_correct_solutions;
}

resolution* prepareValidAlternatives(int num_of_elements_to_insert, int n_factorial, int row, int* h_number_presence, int* h_element_presence, int* h_current_solution)
{
	
	int* numbers_to_insert = defineNumbersToInsert(num_of_elements_to_insert, h_number_presence, row);
	int* positions_to_insert = definePositionsToInsert(num_of_elements_to_insert, h_element_presence, row);

	int** rowPermutations = createPermutations(num_of_elements_to_insert);

	int** alternative_solutions = createAlternativeSolutions(h_current_solution, num_of_elements_to_insert, positions_to_insert, numbers_to_insert, rowPermutations, row);
	int* alternative_solutions_one_array = combineSolutionsIntoOneArray(n_factorial, alternative_solutions);
	bool* alternative_solutions_correctness = checkAlternativeSolutionsCorrectness(row, n_factorial, alternative_solutions_one_array);
	
	return chooseCorrectSolutions(n_factorial, alternative_solutions, alternative_solutions_correctness);	
}

resolution* resolutionArrayIntoStructure(int* solution)
{
	resolution* result_resolution = new resolution();
	int **h_result = new int*[1];

	h_result[0] = solution;
	result_resolution -> n = 1;
	result_resolution -> resolutions = h_result;

	return result_resolution;
}

resolution* createAlternativeSolutions(int row, int* h_current_solution, int* d_current_solution)
{
	int* d_number_presence = fillNumberPresenceInRowsArray(d_current_solution);
	int* d_element_presence = fillElementPresenceInRowsArray(d_current_solution);

	int* h_number_presence = copySudokuToHost(d_number_presence);
	int* h_element_presence = copySudokuToHost(d_element_presence);

	int num_of_elements_to_insert = countEmptyElemsInRow(row, d_number_presence);
	int n_factorial = factorial(num_of_elements_to_insert);

	resolution* alternative_solution;

	if(num_of_elements_to_insert > 0)
		alternative_solution = prepareValidAlternatives(num_of_elements_to_insert, n_factorial, row, h_number_presence, h_element_presence, h_current_solution);
	else
		alternative_solution = resolutionArrayIntoStructure(h_current_solution);

	cudaFree(d_number_presence);
	cudaFree(d_element_presence);
	return alternative_solution;
}

void sumNumberPresenceArray(int* d_number_presence)
{
	dim3 dimBlock2 = dim3(243, 1, 1);
	dim3 dimGrid2 = dim3(1);

	__sumNumberPresenceArray <<<dimGrid2, dimBlock2>>> (d_number_presence, 243);
	cudaErrorHandling(cudaDeviceSynchronize());
}


int* fillNumberPresenceArray(int* d_sudoku) 
{
	int* d_number_presence;
	int sharedMemorySize = 243 * sizeof(int);
	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);

	cudaErrorHandling(cudaMalloc((void **)&d_number_presence, 243 * sizeof(int)));

	__fillFullNumberPresenceArray <<<dimGrid, dimBlock, sharedMemorySize>>> (d_sudoku, d_number_presence);
	cudaErrorHandling(cudaDeviceSynchronize());

	//displayNumberPresenceArray(d_number_presence);

	return d_number_presence;
}

bool defineIfSudokuIsSolved(int* d_number_presence_summed)
{
	int* result = new int[1];

	cudaErrorHandling(cudaMemcpy(result, d_number_presence_summed, sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrorHandling(cudaDeviceSynchronize());

	// printf("---------- FINAL RESULT!!! ------\n");
	// printf("SUMA: %d\n", result[0]);
	if(result[0] == 243)
	{
		// printf("Sudoku jest rozwiązane!\n");
		return true;
	} else
	{
		// printf("Sudoku nie jest rozwiązane! :( \n");
		return false;
	}
}

bool checkIfSudokuIsSolved(int* h_sudoku)
{
	int* d_number_presence;
	bool isSudokuSolved;
	int* d_sudoku = copySudokuToDevice(h_sudoku);

	d_number_presence = fillNumberPresenceArray(d_sudoku);

	sumNumberPresenceArray(d_number_presence);

	isSudokuSolved = defineIfSudokuIsSolved(d_number_presence);

	cudaFree(d_number_presence);

	return isSudokuSolved;
}

int* insertRowToSolution(int row, int* current_solution, int* quiz)
{
	int* solution_copy = duplicateSudoku(current_solution);
	for(int i = 0; i < NN; i ++)
	{
		solution_copy[row*NN + i] = quiz[row*NN + i];
	}
	return solution_copy;
}

resolution* chooseFullyCorrectResolutions(resolution* alternative_solutions)
{
	resolution* correct_resolution = new resolution();
	int k = 0;
	
	correct_resolution -> resolutions = new int*[alternative_solutions -> n];
	for(int i = 0; i < alternative_solutions -> n; i++)
		if(checkIfSudokuIsSolved(alternative_solutions->resolutions[i]))
		{
			solution_found = true;
			correct_resolution -> resolutions[k] = alternative_solutions->resolutions[i];
			k++;
		}

	correct_resolution -> n = k;

	return correct_resolution;
}

resolution* combineResolutionsFromNextRows(resolution* alternative_solutions, int* quiz, int row, bool showProgress)
{
	resolution** next_row_solutions = new resolution*[alternative_solutions -> n];
	
	for(int i = 0; i < alternative_solutions -> n; i ++)
		next_row_solutions[i] = createRowSolutionRecursive(row + 1, alternative_solutions -> resolutions[i], quiz, showProgress);
	
	int alternatives_count = 0;

	for(int i = 0; i < alternative_solutions -> n; i ++)
		alternatives_count += next_row_solutions[i] -> n;


	resolution* final_resolution = new resolution();
	final_resolution -> n = alternatives_count;
	final_resolution -> resolutions = new int*[alternatives_count];
	int k = 0;

	for(int i = 0; i < alternative_solutions -> n; i++)
	{
		for(int j = 0; j < next_row_solutions[i]->n; j++)
		{
			final_resolution->resolutions[k] = next_row_solutions[i]->resolutions[j];
			k++;
		}
	}
	
	return final_resolution;
}

resolution* createRowSolutionRecursive(int row, int* previous_solution, int* quiz, bool showProgress)
{
	int* current_solution, *d_current_solution;
	int sum_empty_elems_in_row;
	resolution* created_resolution = new resolution();

	if(solution_found)
		{
			printf("Oho! Rozwiązanie już znalezione, nic tu po mnie\n");
			created_resolution -> n = 0;
			return created_resolution;
		}

	if(showProgress)
		printf("%d", row + 1);

	current_solution = insertRowToSolution(row, previous_solution, quiz);
	d_current_solution = copySudokuToDevice(current_solution);

	resolution* alternative_solutions = createAlternativeSolutions(row, current_solution, d_current_solution);
	
	cudaFree(d_current_solution);

	if(row == 8)
		return chooseFullyCorrectResolutions(alternative_solutions);
	else
		return combineResolutionsFromNextRows(alternative_solutions, quiz, row, showProgress);
}

void displayResult(resolution* final_resolution)
{
	if (final_resolution -> n == 0)
		printf("Brak prawidłowych rozwiązań zagadki.");
	else
		for(int i = 0; i < final_resolution -> n; i++)
			displayHostArray("POPRAWNE ROZWIĄZANIE", final_resolution -> resolutions[i], NN, NN);
}

cudaError_t solveSudoku(int* h_sudoku_solved, int* h_sudoku_unsolved, bool showProgress)
{
  int* empty_resolution = newArrayWithZero(NN, NN);

  displayHostArray("SUDOKU QUIZ", h_sudoku_unsolved, NN, NN);

	resolution* final_resolution = createRowSolutionRecursive(0, empty_resolution, h_sudoku_unsolved, showProgress);

	displayResult(final_resolution);

	return cudaSuccess;
}

