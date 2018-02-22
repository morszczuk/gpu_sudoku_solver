#include "kernel.h"

void cudaErrorHandling(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error on CUDA %d: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
	}
}

__global__ void __scan(int *g_odata, int *g_idata, int n)
{
  extern __shared__ int temp[]; // allocated on invocation
  // int thid = threadIdx.x;
	// int thid = blockDim.x*blockIdx.x + threadIdx.x;
	int thid = blockDim.y*blockIdx.y + threadIdx.y;
	int blockOffset = blockDim.x*blockIdx.x + threadIdx.x;

	printf("THID: %d, blockOffset: %d\n", thid, blockOffset);
  
	int pout = 0, pin = 1;
  // load input into shared memory.
  // This is exclusive scan, so shift right by one and set first elt to 0
  // temp[pout*n + thid] = g_idata[thid];
	temp[pout*n + thid + blockOffset*9] = (thid > 0) ? g_idata[thid-1 + blockOffset*9] : 0;
	// temp[pin*n + thid] = temp[pout*n + thid];
	temp[pin*n + thid + blockOffset*9] = 0;
	// printf("THID: %d, g_idata: %d\n", thid, temp[pout*n + thid + blockOffset*9]);
  
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2)
  {
    pout = 1 - pout; // swap double buffer indices
    pin = 1 - pout;
    if (thid >= offset)
		{
			// printf("THID>OFFSET, THID: %d, id1: %d, id2: %d, %d + %d = %d\n", thid, pout*n+thid, pin*n+thid - offset, temp[pout*n+thid], temp[pin*n+thid - offset], temp[pout*n+thid] + temp[pin*n+thid - offset]);
      temp[pout*n+thid + blockOffset*9] = temp[pin*n+thid + blockOffset*9] + temp[pin*n+thid - offset + blockOffset*9];
		}
    else
      temp[pout*n+thid + blockOffset*9] = temp[pin*n+thid + blockOffset*9];

    __syncthreads();
		//printf("THID: %d, OFFSET: %d, TEMP[%d]: %d\n", thid, offset, pout*n + thid, temp[pout*n + thid]);
		__syncthreads();
  }

	if(thid > 0)
  	g_odata[thid-1+ blockOffset*9] = temp[pout*n+thid+ blockOffset*9]; // write output
	if (thid == n -1 )
		g_odata[thid+ blockOffset*9] = g_odata[thid-1+ blockOffset*9] + g_idata[thid+ blockOffset*9];
} 


__global__ void __prescan(int *g_odata, int *g_idata, int n)
{
  extern __shared__ int temp[];// allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;
	printf("THID: [%d]\n", thid);

	// printf("---\ntemp[2*%d] = %d\ntemp[2*%d+1] =%d\n", thid, temp[2*thid], thid, temp[2*thid+1]);
  temp[2*thid] = g_idata[2*thid]; // load input into shared memory
  
	if(thid < 4)
	{
		// printf("2*thid %d\n", thid);
		temp[2*thid+1] = g_idata[2*thid+1];
	}

	__syncthreads();


  for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d)
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }


  // if (thid == 0) { temp[n - 1] = 0; } // clear the last element


  // for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  // {
  //   offset >>= 1;
  //   __syncthreads();
  //   if (thid < d)
  //   {
  //     int ai = offset*(2*thid+1)-1;
  //     int bi = offset*(2*thid+2)-1;
  //     int t = temp[ai];
  //     temp[ai] = temp[bi];
  //     temp[bi] += t;
  //   }
  // }


  // __syncthreads();


  // g_odata[2*thid] = temp[2*thid]; // write results to device memory
  // g_odata[2*thid+1] = temp[2*thid+1];
} 

bool defineIfSudokuIsSolved(int* d_number_presence_summed)
{
	int* result = new int[1];

	cudaErrorHandling(cudaMemcpy(result, d_number_presence_summed, sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrorHandling(cudaDeviceSynchronize());

	printf("---------- FINAL RESULT!!! ------\n");
	printf("SUMA: %d\n", result[0]);
	if(result[0] == 243)
	{
		printf("Sudoku jest rozwiązane!\n");
		return true;
	} else
	{
		printf("Sudoku nie jest rozwiązane! :( \n");
		return false;
	}
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

void sumNumberPresenceArray(int* d_number_presence)
{
	dim3 dimBlock2 = dim3(243, 1, 1);
	dim3 dimGrid2 = dim3(1);

	__sumNumberPresenceArray <<<dimGrid2, dimBlock2>>> (d_number_presence, 243);
	cudaErrorHandling(cudaDeviceSynchronize());
}


void displayNumberPresenceArray(int* d_number_presence)
{
	int* h_number_presence = new int[243];

	cudaErrorHandling(cudaMemcpy(h_number_presence, d_number_presence, 243 * sizeof(int), cudaMemcpyDeviceToHost));

	printf("---------NUMBER PRESENCE ARRAY-----------\n");
	for (int i = 0; i < 27; i++)
	{
		for(int j = 0; j < 9; j++)
		{
			printf("%d |", h_number_presence[i*9 + j]);
		}
		printf("\n");
	}
	printf("-----------------------------------------\n");
}

void displaySudokuArray(int* d_number_presence)
{
	int* h_number_presence = new int[81];

	cudaErrorHandling(cudaMemcpy(h_number_presence, d_number_presence, 81 * sizeof(int), cudaMemcpyDeviceToHost));

	printf("---------NUMBER PRESENCE IN ROW-----------\n");
	for (int i = 0; i < 9; i++)
	{
		for(int j = 0; j < 9; j++)
		{
			printf("%d |", h_number_presence[i*9 + j]);
		}
		printf("\n");
	}
	printf("-----------------------------------------\n");
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

int* fillNumberPresenceArray(int* d_sudoku) 
{
	int* d_number_presence;
	int sharedMemorySize = 243 * sizeof(int);
	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);

	cudaErrorHandling(cudaMalloc((void **)&d_number_presence, 243 * sizeof(int)));

	__fillNumberPresenceArray <<<dimGrid, dimBlock, sharedMemorySize>>> (d_sudoku, d_number_presence);
	cudaErrorHandling(cudaDeviceSynchronize());

	//displayNumberPresenceArray(d_number_presence);

	return d_number_presence;
}

bool checkIfSudokuIsSolved(int* d_sudoku)
{
	int* d_number_presence;
	bool isSudokuSolved;

	d_number_presence = fillNumberPresenceArray(d_sudoku);

	sumNumberPresenceArray(d_number_presence);

	isSudokuSolved = defineIfSudokuIsSolved(d_number_presence);

	return isSudokuSolved;
}

__global__ void __defineNumberPresenceInRow(int* d_quiz_unsolved, int* d_number_presence_in_row)
{
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idy = blockDim.x*blockIdx.x + threadIdx.x;

	if(d_quiz_unsolved[idx*SUD_SIZE + idy] > 0)
	{
		d_number_presence_in_row[idx * SUD_SIZE + d_quiz_unsolved[idx*SUD_SIZE + idy] - 1] = 1;
	} else
	{
		d_number_presence_in_row[idx * SUD_SIZE + d_quiz_unsolved[idx*SUD_SIZE + idy] - 1] = 0;
	}
}

int* defineNumberPresenceInRow(int* d_quiz_unsolved)
{
	int *d_number_presence_in_row;
	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);

	cudaErrorHandling(cudaMalloc((void **)&d_number_presence_in_row, SUD_SIZE * SUD_SIZE * sizeof(int)));

	__defineNumberPresenceInRow <<<dimGrid, dimBlock>>>(d_quiz_unsolved, d_number_presence_in_row);
	cudaErrorHandling(cudaDeviceSynchronize());
	printf("\n\nPO DEFINE\n\n\n");
	displaySudokuArray(d_number_presence_in_row);

	return d_number_presence_in_row;
}

int* scanNumberPresenceInRow(int* d_number_presence_in_row)
{
	int *d_scanned_number_presence_in_row;
	dim3 dimBlock = dim3(9, 9, 1);
	dim3 dimGrid = dim3(1);
	int sharedMemorySize = 18*9* sizeof(int);

	printf("\n\n\nPRZED ALOKACJĄ\n\n\n");

	cudaErrorHandling(cudaMalloc((void **)&d_scanned_number_presence_in_row, SUD_SIZE * SUD_SIZE * sizeof(int)));

	printf("\n\n\nPRZED SCANEM\n\n\n");
	//__prescan <<<dimGrid, dimBlock, sharedMemorySize>>> (d_scanned_number_presence_in_row, d_number_presence_in_row, 9);
	__scan <<<dimGrid, dimBlock, sharedMemorySize>>> (d_scanned_number_presence_in_row, d_number_presence_in_row, 9);
	
	cudaErrorHandling(cudaDeviceSynchronize());

	displaySudokuArray(d_scanned_number_presence_in_row);
	return d_scanned_number_presence_in_row;
}

int* createSolution(int* d_quiz_unsolved)
{
	int *d_number_presence_in_row;
	int *d_scanned_number_presence_in_row;

	d_number_presence_in_row = defineNumberPresenceInRow(d_quiz_unsolved);
	printf("\n\nWSKAZNIK PRESENCE ZWROCONY\n\n\n");
	d_scanned_number_presence_in_row = scanNumberPresenceInRow(d_number_presence_in_row);
	

}

cudaError_t solveSudoku(int* h_sudoku_quiz_solved, int* h_sudoku_quiz_unsolved)
{
	int *d_sudoku_quiz_unsolved, *d_sudoku_quiz_solved, *d_quiz_fill;	
	int i = 0;

	cudaErrorHandling(cudaMalloc((void **)&d_sudoku_quiz_unsolved, SUD_SIZE * SUD_SIZE * sizeof(int)));
	cudaErrorHandling(cudaMalloc((void **)&d_sudoku_quiz_solved, SUD_SIZE * SUD_SIZE * sizeof(int)));

	cudaErrorHandling(cudaMemcpy(d_sudoku_quiz_unsolved, h_sudoku_quiz_unsolved, SUD_SIZE * SUD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	cudaErrorHandling(cudaMemcpy(d_sudoku_quiz_solved, h_sudoku_quiz_solved, SUD_SIZE * SUD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
	
	d_quiz_fill = d_sudoku_quiz_unsolved;
	while(!checkIfSudokuIsSolved(d_quiz_fill))
	{
		if( i > 5)
			d_quiz_fill = d_sudoku_quiz_solved;
		else
		{
			cudaErrorHandling(cudaMemcpy(d_sudoku_quiz_unsolved, h_sudoku_quiz_unsolved, SUD_SIZE * SUD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
			d_quiz_fill = d_sudoku_quiz_unsolved;
			createSolution(d_sudoku_quiz_unsolved);
		}
		i++;
	}


	return cudaSuccess;
}