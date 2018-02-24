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

int* insertRowToSolution(int row, int* current_solution, int* quiz)
{
	int* solution_copy = duplicateSudoku(current_solution);
	for(int i = 0; i < NN; i ++)
	{
		solution_copy[row*NN + i] = quiz[row*NN + i];
	}
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

int countEmptyElemsInRow(int row, int* d_current_solution)
{
	int* d_number_presence = fillNumberPresenceInRowsArray(d_current_solution);
	int filled_elements = sumNumberPresenceInRow(d_number_presence, row);

	printf("LICZBA ELEMENTÓW WYPEŁNIONYCH w rzędzie %d: %d\n", row + 1, filled_elements);

	return NN - filled_elements;
}

void createPermutations(int empty_elems_in_row)
{
	int test[empty_elems_in_row];
	int myints[] = {1,2,3};

	for(int i = 0; i < empty_elems_in_row; i ++)
	{
		test[i] = i;
	}

	printf("PERMUTUJEMY\n");

	do
	{
		// printf("%d | %d | %d\n", myints[0], myints[1], myints[2]);
		// printf("%d | %d | %d\n", myints[0], myints[1], myints[2]);
		for(int i = 0; i < empty_elems_in_row; i++)
		{
			printf("%d | ", test[i]);
		}
		printf("\n");
	} while (std::next_permutation(test,test+empty_elems_in_row));
}

resolution* createRowSolution(int row, int* _current_solution, int* quiz)
{
	int* current_solution, *d_current_solution;
	int empty_elems_in_row;
	resolution* created_resolution = new resolution();

	current_solution = insertRowToSolution(row, _current_solution, quiz);
	
	d_current_solution = copySudokuToDevice(current_solution);

	empty_elems_in_row = countEmptyElemsInRow(row, d_current_solution);

	createPermutations(empty_elems_in_row);

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
	printf("Wynikow: %d\n", final_resolution -> n);

	return cudaSuccess;
}

