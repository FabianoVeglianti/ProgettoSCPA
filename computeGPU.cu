#include <cuda_runtime.h>
#include "computeGPU.h"
#include <stdio.h>


#define NZ_PER_BLOCK 32

__global__ void csr_scalar(int M,
								const int *JA, 
								const int *IRP, 
								const double *AS,
								const double *val_vec,
								double *val_res)
{
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < M){
		int row_start = IRP[row];
		int row_end = IRP[row+1];
		
		double sum = 0;
		for(int i = row_start; i < row_end; i++)
		{
			sum += AS[i] * val_vec[JA[i]];
		}
		val_res[row] = sum;
	}
	
}

__device__ double warp_reduce(double value){
	for(int offset = 16; offset > 0; offset /=2)
		value += __shfl_down_sync(0xffffffff, value, offset);
	
	return value;
}

__global__ void csr_vector(int M,
								const int *JA, 
								const int *IRP, 
								const double *AS,
								const double *val_vec,
								double *val_res)
{
	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = thread_id / 32; //uguale a warp_id
	const int lane = thread_id % 32;
	
	double sum = 0;
	if(row < M)
	{
		int row_start = IRP[row];
		int row_end = IRP[row+1];
		
		for(int i = row_start + lane; i < row_end; i+= 32)
		{
			sum += AS[i] * val_vec[JA[i]];
		}
	
	
	
		sum = warp_reduce(sum);
		
		if(lane == 0)
			val_res[row] = sum;
	}
}

int create_row_blocks(bool onlyCount, int M, const int *IRP,int *row_blocks)
/* Implementazione dell'algoritmo 2 del paper "Efficient Sparse Matrix-Vector Mulitplication on GPUs using CSR Storage Format"
 * Dal momento che non possiamo sapere a priori il numero di blocchi di righe che si verranno a creare
 * l'algortmo viene eseguito 2 volte: nella prima volta onlyCount è true e quindi tutti gli accessi a
 * row_blocks vengono omessi, quindi si contano i blocchi che si verrebbero a stabilire; la seconda volta 
 * l'algoritmo viene eseguito con onlyCount a false e quindi si stabiliscono i blocchi di righe.*/
{
	if(!onlyCount){
		row_blocks[0] = 0;
	}
	int last_i = 0; //ultima riga inserita in un blocco di righe
	int current_block = 1;
	int nz = 0; //nz che sto cercando di inserire nel blocco di righe
	for(int i = 1; i <= M; i++)
	{
		nz += IRP[i] - IRP[i - 1];

		if(nz == NZ_PER_BLOCK)
		{
			//quesa riga riempie esattamente fino a NZ_PER_BLOCK
			last_i = i;

			if (!onlyCount){
				row_blocks[current_block] = i;
			}
			current_block++;
			nz = 0;
		}
		else if (nz > NZ_PER_BLOCK)
		{
			//la riga considerata non entra nel blocco corrente di righe
			if (i - last_i > 1)
			{
				//se il blocco di righe che sto considerando vorrebbe contenere più di una riga chiudo il blocco alla riga precedente e riconsidero la riga
				if (!onlyCount){
					row_blocks[current_block] = i - 1;
				}
				current_block++;
				i--;
			}
			else
			{
				//il blocco di righe che sto considerando vorrebbe contenere un'unica riga -> non posso far altro che avere un blocco di righe che contiene più di 32 elementi
				// in csr_adaptive verrà gestito da csr_vector
				if (!onlyCount){
					row_blocks[current_block] = i;
				}
				current_block++;
			}

			last_i = i;
			nz = 0;
		}
		//se ci sono NZ_PER_BLOCK righe vuote, devo creare un blocco di righe senza tener conto del numero di nz
		else if (i - last_i > NZ_PER_BLOCK)
		{
			last_i = i;
			if (!onlyCount){
				row_blocks[current_block] = i;
			}
			current_block++;
			nz = 0;
		}
	}

	if (!onlyCount)
		row_blocks[current_block] = M;

	return current_block;
}

__device__ int prev_power_of_two(int n)
{
	if(n == 0)
		return 0;
	int m = 1;
	while (n/2>=1){
		n /= 2;
		m *=2;
	}
	return m;
}


__global__ void csr_adaptive(const int M, 
								const int *JA, 
								const int *IRP, 
								const double *AS, 
								const double *vec_val,
								double *res_val,
								const int *row_blocks){
	const int block_row_start = row_blocks[blockIdx.x];
	const int block_row_end = row_blocks[blockIdx.x + 1];
	
	
	if (block_row_end - block_row_start > 1)
	{
    // CSR-Stream 
    
    __shared__ double sharedmem[NZ_PER_BLOCK];
    const int nz = IRP[block_row_end] - IRP[block_row_start];
    
    
   const int i = threadIdx.x;
	const int block_data_begin = IRP[block_row_start];
	const int thread_data_begin = block_data_begin + i;

	if (i < nz)
		sharedmem[i] = AS[thread_data_begin] * vec_val[JA[thread_data_begin]];
	__syncthreads ();

	const int threads_for_reduction = prev_power_of_two(blockDim.x / (block_row_end - block_row_start));

	if (threads_for_reduction > 1)
		{
		//Riduzione dei nz ad opera di piu' threads
		const int thread_in_tfr = i % threads_for_reduction; //indice del thread in threads_for_reduction
		const int local_row = block_row_start + i / threads_for_reduction; //indice della riga in IRP che il thread i deve contribuire a ridurre

		double sum = 0.0;
		
		//local_row potrebbe non appartenere a questo blocco
		if (local_row < block_row_end) 
			{
			//indice del primo elemento della riga local_row nel blocco di righe
			const int local_first_element = IRP[local_row] - IRP[block_row_start]; 
			//indice dell'ultimo elemento della riga local_row nel blocco di righe + 1
			const int local_last_element = IRP[local_row + 1] - IRP[block_row_start];

		//ciascuno dei threads_for_reduction thread somma in sum i risultati parziali della riga in sharedmem a salti di threads_for_reduction
			for (int local_element = local_first_element + thread_in_tfr; local_element < local_last_element; local_element += threads_for_reduction)
				sum += sharedmem[local_element];
			}
		__syncthreads ();
		//dunque ciascun thread salva il valore di sum in sharedmem, cosi' ci sono threads_for_reduction (potenza di 2) valori da ridurre.
		sharedmem[i] = sum;

		//ogni riga ha threads_for_reduction elementi in sharedmem da ridurre
		for (int j = threads_for_reduction / 2; j > 0; j /= 2)
		{
			//riduzione
			__syncthreads ();

			//non tutti i threads devono scrivere
			//la prima proposizione prende solo la prima meta' dei threads
			//la seconda proposizione evita di uscire dalla sharedmem
			const bool use_result = thread_in_tfr < j && i + j < NZ_PER_BLOCK;

			if (use_result)
				sum += sharedmem[i + j];
			__syncthreads ();

			if (use_result)
			  sharedmem[i] = sum;
		}
		//il thread 0 salva il valore trovato
		if(thread_in_tfr == 0 && local_row < block_row_end)
			res_val[local_row] = sum;
		}
		else
		{
		//riduzione di tutti i nz ad opera di un singolo threads
			int local_row = block_row_start + i;
			while (local_row < block_row_end)
			{
				double sum = 0.0;

				for ( int j = IRP[local_row] - block_data_begin; j < IRP[local_row + 1] - block_data_begin;	j++)
					sum += sharedmem[j];

				res_val[local_row] = sum;
				local_row += NZ_PER_BLOCK;
			}
		}
	} 
	else 
	{
		// CSR-Vector
		const int warp_id = threadIdx.x / 32;
		const int lane = threadIdx.x % 32;
		const int row = block_row_start;
		

		double sum = 0;
		if(row < M)
		{
			const int row_start = IRP[row];
			const int row_end = IRP[row + 1];

			for(int i = row_start + lane; i < row_end; i += 32)
				sum += AS[i] * vec_val[JA[i]];
		

			sum = warp_reduce(sum);

			if (lane == 0 && warp_id ==0)
			{
				res_val[row] = sum;
			}
			
		}
	}
	
}


__global__ void ellpack_kernel(int M, 
									int max_nz, 
									const int *JA,
									const double *AS,
									const double *val_vec,
									double *val_res)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < M){
		double sum = 0;
		for(int i = 0; i < max_nz; i++){
			const int index = M * i + row; //tiene conto della trasposizione della matrice  
			sum += AS[index] * val_vec[JA[index]];
		}
		val_res[row] = sum;
	}
}
