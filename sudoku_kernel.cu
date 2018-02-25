#include "sudoku_kernel.h"

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
	int k = SUD_SIZE*SUD_SIZE;

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
	int k = SUD_SIZE*SUD_SIZE;

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

int* insertRowToSolution(int row, int* current_solution, int* quiz)
{
	printf("TUTAJ DOJDZIEMY? -1.0.1\n");
	int* solution_copy = duplicateSudoku(current_solution);
	printf("TUTAJ DOJDZIEMY? -1.0.2\n");
	for(int i = 0; i < NN; i ++)
	{
		solution_copy[row*NN + i] = quiz[row*NN + i];
	}
	printf("TUTAJ DOJDZIEMY? -1.0.3\n");
	return solution_copy;
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
			printf("O, dodaję element!!! Liczba do wstawienia: %d\n", (j % NN) + 1);
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
			printf("Pozycja do wstawienia: %d\n", j % NN);
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

	printf("LICZBA ELEMENTÓW WYPEŁNIONYCH w rzędzie %d: %d\n", row + 1, filled_elements);

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

__device__ void __sumNumberPresence(int* d_number_presence_in_col, int size)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	// printf("[IDX: %d]\n", idx);
	__syncthreads();

	for (int i = 1; i <= size / 2; i *= 2)
	{
		if (threadIdx.x % (2 * i) == 0) {
			printf("BEFORE [Thread %d]: %d\n", idx, d_number_presence_in_col[idx]);
			d_number_presence_in_col[idx] += d_number_presence_in_col[idx + i];
			printf("AFTER [Thread %d]: %d\n", idx, d_number_presence_in_col[idx]);
		}
		else
		{
			printf("[Thread %d] returning\n", idx);
			return;
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
	{
		d_number_presence_in_col[idx] += d_number_presence_in_col[idx + 64];
		printf("DODAŁEM! WYNIK: %d\n", d_number_presence_in_col[idx]);
	}

}

__global__ void __checkAlternativeSolutionsCorrectness(int* d_alternative_solutions_one_array, bool* d_alternative_solutions_correctness, int* d_number_presence_in_col)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int row = threadIdx.x % NN;
	int col = threadIdx.x - ((threadIdx.x / NN)*NN);
	int rowStart = blockDim.x*blockIdx.x + row*NN;
	int blockStart = blockDim.x*blockIdx.x;

	// printf("Moje IDX: %d", idx);

	d_number_presence_in_col[idx] = 0;

	if(threadIdx.x == 0)
		d_alternative_solutions_correctness[blockIdx.x] = false;

	__syncthreads();

	printf("IDX: %d | WARTOSC: %d | COL: %d | INDEKS DO WSTAIWENIA: %d\n", idx, d_alternative_solutions_one_array[idx], col, blockStart + (col * NN) + d_alternative_solutions_one_array[idx] - 1);
	if(d_alternative_solutions_one_array[idx] > 0)
	{
		printf("AKTUALNA WARTOSC: %d\n", d_number_presence_in_col[blockStart + (col * NN) + d_alternative_solutions_one_array[idx] - 1]);
		d_number_presence_in_col[blockStart + (col * NN) + d_alternative_solutions_one_array[idx] - 1] += 1; //informs about number data[idx][idy] - 1 presence in column idy
	}
	//number_presence[k + (idy * SUD_SIZE + d_sudoku[idx*SUD_SIZE + idy] - 1)] = 1; //informs about number data[idx][idy] - 1 presence in column idy
	//d_number_presence_in_row[rowStart + d_alternative_solutions_one_array[idx] - 1] += 1;

	__syncthreads();

	if(d_number_presence_in_col[idx] > 1)
		d_alternative_solutions_correctness[blockIdx.x] = true;

}

bool** checkAlternativeSolutionsCorrectness(int n_factorial, int* alternative_solutions_one_array)
{
	int* d_alternative_solutions_one_array = copyArrayToDevice(alternative_solutions_one_array, n_factorial * NN * NN);
	int* d_number_presence_in_row;
	bool* d_alternative_solutions_correctness, *h_alternative_solutions_correctness;
	h_alternative_solutions_correctness = new bool[n_factorial];
	
	
	cudaErrorHandling(cudaMalloc((void **)&d_alternative_solutions_correctness, n_factorial * sizeof(bool)));
	cudaErrorHandling(cudaMalloc((void **)&d_number_presence_in_row, n_factorial * NN * NN * sizeof(int)));

	dim3 dimBlock = dim3(81, 1, 1);
	dim3 dimGrid = dim3(n_factorial);

	__checkAlternativeSolutionsCorrectness <<<dimGrid, dimBlock>>>(d_alternative_solutions_one_array, d_alternative_solutions_correctness, d_number_presence_in_row);
	cudaErrorHandling(cudaDeviceSynchronize());

	cudaErrorHandling(cudaMemcpy(h_alternative_solutions_correctness, d_alternative_solutions_correctness, n_factorial * sizeof(bool), cudaMemcpyDeviceToHost));
	for(int i = 0; i < n_factorial; i++)
	{
		printf("\nRozwiazanie %d: ", i);
		if(h_alternative_solutions_correctness[i])
			printf("ZLE\n");
		else
			printf("OK\n");
	}
}

int** createAlternativeSolutions(int row, int* h_current_solution, int* d_current_solution)
{
	int* d_number_presence = fillNumberPresenceInRowsArray(d_current_solution);
	int* d_element_presence = fillElementPresenceInRowsArray(d_current_solution);

	int* h_number_presence = copySudokuToHost(d_number_presence);
	int* h_element_presence = copySudokuToHost(d_element_presence);
	int num_of_elements_to_insert = countEmptyElemsInRow(row, d_number_presence);
	if(num_of_elements_to_insert > 0)
	{
		int n_factorial = factorial(num_of_elements_to_insert);
		int* numbers_to_insert = defineNumbersToInsert(num_of_elements_to_insert, h_number_presence, row);
		int* positions_to_insert = definePositionsToInsert(num_of_elements_to_insert, h_element_presence, row);
		int** rowPermutations = createPermutations(num_of_elements_to_insert);
		int** alternative_solutions = createAlternativeSolutions(h_current_solution, num_of_elements_to_insert, positions_to_insert, numbers_to_insert, rowPermutations, row);
		int* alternative_solutions_one_array = combineSolutionsIntoOneArray(n_factorial, alternative_solutions);
		bool** alternative_solutions_correctness = checkAlternativeSolutionsCorrectness(n_factorial, alternative_solutions_one_array);
		return alternative_solutions;
	} else
	{
		int **h_result = new int*[1];
		h_result[0] = h_current_solution;
		return h_result;
	}
}

resolution* createRowSolution(int row, int* _current_solution, int* quiz)
{
	int* current_solution, *d_current_solution;
	int sum_empty_elems_in_row;
	resolution* created_resolution = new resolution();
	printf("TUTAJ DOJDZIEMY? -1\n");

	current_solution = insertRowToSolution(row, _current_solution, quiz);
	printf("TUTAJ DOJDZIEMY? -1.1\n");
	d_current_solution = copySudokuToDevice(current_solution);
	printf("TUTAJ DOJDZIEMY? -1.2\n");
	int** alternative_solutions = createAlternativeSolutions(row, current_solution, d_current_solution);
	// bool** alternative_solutions_correctness = checkAlternativeSolutionsCorrectness(alternative_solutions);
	current_solution = alternative_solutions[0];
	printf("TUTAJ DOJDZIEMY? 7\n");
	// sum_empty_elems_in_row = countEmptyElemsInRow(row, d_current_solution);

	// createPermutations(sum_empty_elems_in_row);

	if(row == 8)
	{
		printf("TUTAJ DOJDZIEMY? 8\n");
		created_resolution -> n = 1;
		created_resolution -> resolutions = current_solution;
		return created_resolution;
	} else
	{
		printf("TUTAJ DOJDZIEMY? 9\n");
		return createRowSolution(row + 1, current_solution, quiz);
	}
}

cudaError_t solveSudoku(int* h_sudoku_solved, int* h_sudoku_unsolved)
{
  int* empty_resolution = newArrayWithZero(NN, NN);
	resolution* final_resolution;
  displayHostArray("SUDOKU QUIZ", h_sudoku_unsolved, NN, NN);
	displayHostArray("RESOLUTION", empty_resolution, NN, NN);

	final_resolution = createRowSolution(0, empty_resolution, h_sudoku_unsolved);
	printf("Wynikow: %d\n", final_resolution -> n);

	return cudaSuccess;
}

