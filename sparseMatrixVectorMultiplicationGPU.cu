#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>
extern "C"{
#include "./IO/mmio.h"
#include "./IO/io.h"
#include "./IO/utils.h"
#include "compute.h"
}
#include "computeGPU.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>

#define EXECUTIONS 32
#define MATRICES_FOLDER "/data/matrici/"
#define MATRICES_FILE "matrici.txt"
#define BLOCK_SIZE_CSR_VECTOR 256
#define BLOCK_SIZE_CSR_SCALAR 256
#define BLOCK_SIZE_ELLPACK 256
#define NZ_PER_BLOCK 32
#define TOLERANCE 10E-7

extern "C"
double csr_scalar_cuda(struct csrFormat *csr, struct vector *vec, struct vector *res, int nz){
	int *d_JA, *d_IRP;
	double *d_AS, *d_val_vec, *d_val_res;
	
	int *IRP = csr->IRP;
	int *JA = csr->JA;
	double *AS = csr->AS;
	int M = csr->M;
	int N = csr->N;
	double *res_val = (double *) malloc(M * sizeof(double));
	
	const dim3 block_size = dim3(BLOCK_SIZE_CSR_SCALAR);
	const dim3 grid_size = dim3((M + BLOCK_SIZE_CSR_SCALAR - 1)/BLOCK_SIZE_CSR_SCALAR); 
	
	
	//allocazione memoria sul device e invio dei dati
	checkCudaErrors(cudaMalloc((void **)&d_IRP, (M+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_JA, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_AS, nz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_vec, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_res, M*sizeof(double)));

	
	checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M+1)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_JA, JA, nz*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_AS, AS, nz*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_val_vec, vec->val, N*sizeof(double), cudaMemcpyHostToDevice));

	double elapsed_time = 0.0;
	for(int i = 0; i < EXECUTIONS; i++){
		cudaEvent_t start,stop;
		
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		cudaEventRecord(start, 0);
		
		csr_scalar<<<grid_size, block_size>>>(M, d_JA, d_IRP, d_AS, d_val_vec, d_val_res);
		
		cudaEventRecord(stop,0);
		checkCudaErrors(cudaEventSynchronize(stop));
		
		float time;
		checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
		
		elapsed_time += (double) time;
		
		if(i == 0){
			checkCudaErrors(cudaMemcpy(res_val, d_val_res, M*sizeof(double), cudaMemcpyDeviceToHost));
			res->val = res_val;
			res->dim = M;
		}
	}
	
	checkCudaErrors(cudaFree(d_IRP));
	checkCudaErrors(cudaFree(d_JA));
	checkCudaErrors(cudaFree(d_AS));
	checkCudaErrors(cudaFree(d_val_vec));
	checkCudaErrors(cudaFree(d_val_res));

	

	checkCudaErrors(cudaDeviceSynchronize());
	return elapsed_time;
}
extern "C"
double csr_vector_cuda(struct csrFormat *csr, struct vector *vec, struct vector *res, int nz){
	int *d_JA, *d_IRP;
	double *d_AS, *d_val_vec, *d_val_res;
	

	int *IRP = csr->IRP;
	int *JA = csr->JA;
	double *AS = csr->AS;
	int M = csr->M;
	int N = csr->N;
	double *res_val = (double *) malloc(M * sizeof(double));
		

	const dim3 block_size = dim3 (BLOCK_SIZE_CSR_VECTOR);
	const dim3 grid_size = dim3 ((32*M + BLOCK_SIZE_CSR_VECTOR - 1)/BLOCK_SIZE_CSR_VECTOR); 
	
	//allocazione memoria sul device e invio dei dati
	//checkCudaErrors(cudaMalloc((void **)d_M, 1*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_IRP, (M+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_JA, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_AS, nz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_vec, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_res, M*sizeof(double)));

	
	checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M+1)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_JA, JA, nz*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_AS, AS, nz*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_val_vec, vec->val, N*sizeof(double), cudaMemcpyHostToDevice));

	double elapsed_time = 0.0;
	for(int i = 0; i < EXECUTIONS; i++){
		cudaEvent_t start,stop;
		
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		cudaEventRecord(start, 0);
		
		csr_vector<<<grid_size, block_size>>>(M, d_JA, d_IRP, d_AS, d_val_vec, d_val_res);
		
		cudaEventRecord(stop,0);
		checkCudaErrors(cudaEventSynchronize(stop));
		
		float time;
		checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
		
		elapsed_time += (double)time;
		
		if(i == 0){
			checkCudaErrors(cudaMemcpy(res_val, d_val_res, M*sizeof(double), cudaMemcpyDeviceToHost));
			res->val = res_val;
			res->dim = M;
		}
	}
	
	checkCudaErrors(cudaFree(d_IRP));
	checkCudaErrors(cudaFree(d_JA));
	checkCudaErrors(cudaFree(d_AS));
	checkCudaErrors(cudaFree(d_val_vec));
	checkCudaErrors(cudaFree(d_val_res));

	

	return elapsed_time;
}
extern "C"
double csr_adaptive_cuda(struct csrFormat *csr, struct vector *vec, struct vector *res, int nz){
	int *d_JA, *d_IRP, *d_row_blocks;
	double *d_AS, *d_val_vec, *d_val_res;
	

	int *IRP = csr->IRP;
	int *JA = csr->JA;
	double *AS = csr->AS;
	int M = csr->M;
	int N = csr->N;
	int *row_blocks;
	double *res_val = (double *) malloc(M * sizeof(double));

	int block_count = create_row_blocks(true, M, IRP, row_blocks);
	row_blocks = (int *) malloc((block_count+1) * sizeof(int));
	if(block_count != create_row_blocks(false, M, IRP, row_blocks)){
		fprintf(stderr, "Errore nella creazione dei blocchi di righe.\nFine programma.\n");
		exit(EXIT_FAILURE);
	}

	//calcolo il numero di blocchi necessari a coprire tutti i blocchi di righe
	const dim3 block_size = dim3 (NZ_PER_BLOCK);
	const dim3 grid_size = dim3 (block_count); 
	
	
	//allocazione memoria sul device e invio dei dati
	//checkCudaErrors(cudaMalloc((void **)d_M, 1*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_IRP, (M+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_JA, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_AS, nz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_vec, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_res, M*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_row_blocks, (block_count+1)*sizeof(int)));
	
	checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M+1)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_JA, JA, nz*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_AS, AS, nz*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_val_vec, vec->val, N*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_row_blocks, row_blocks, (block_count+1)*sizeof(int), cudaMemcpyHostToDevice));

	double elapsed_time = 0.0;
	for(int i = 0; i < EXECUTIONS; i++){
		cudaEvent_t start,stop;
		
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		cudaEventRecord(start, 0);
		
		csr_adaptive<<<grid_size, block_size>>>(M, d_JA, d_IRP, d_AS, d_val_vec, d_val_res, d_row_blocks);
		
		cudaEventRecord(stop,0);
		checkCudaErrors(cudaEventSynchronize(stop));
		
		float time;
		checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
		
		elapsed_time += (double)time;
		
		if(i == 0){
			checkCudaErrors(cudaMemcpy(res_val, d_val_res, M*sizeof(double), cudaMemcpyDeviceToHost));
			res->val = res_val;
			res->dim = M;
		}
	}
	
	checkCudaErrors(cudaFree(d_IRP));
	checkCudaErrors(cudaFree(d_JA));
	checkCudaErrors(cudaFree(d_AS));
	checkCudaErrors(cudaFree(d_val_vec));
	checkCudaErrors(cudaFree(d_val_res));
	checkCudaErrors(cudaFree(d_row_blocks));
	free(row_blocks);
	
	return elapsed_time;
}

extern "C"
double ellpack_cuda(struct ellpackFormat *ellpack, struct vector *vec, struct vector *res, int nz){
	int *d_JA;
	double *d_AS, *d_val_vec, *d_val_res;
	

	
	int *JA = ellpack->JA;
	double *AS = ellpack->AS;
	int M = ellpack->M;
	int N = ellpack->N;
	int maxnz = ellpack->MAXNZ;
	double *res_val = (double *) malloc(M * sizeof(double));
	
		

	const dim3 block_size = dim3 (BLOCK_SIZE_ELLPACK);
	const dim3 grid_size = dim3 ((M + BLOCK_SIZE_ELLPACK - 1)/BLOCK_SIZE_ELLPACK); 
	
	//allocazione memoria sul device e invio dei dati
	//checkCudaErrors(cudaMalloc((void **)d_M, 1*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_JA, maxnz*M*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_AS, maxnz*M*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_vec, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_val_res, M*sizeof(double)));

	
	checkCudaErrors(cudaMemcpy(d_JA, JA, maxnz*M*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_AS, AS, maxnz*M*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_val_vec, vec->val, N*sizeof(double), cudaMemcpyHostToDevice));


	double elapsed_time = 0.0;
	for(int i = 0; i < EXECUTIONS; i++){
		cudaEvent_t start,stop;
		
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		cudaEventRecord(start, 0);
		
		ellpack_kernel<<<grid_size, block_size>>>(M, maxnz, d_JA, d_AS, d_val_vec, d_val_res);
		
		cudaEventRecord(stop,0);
		checkCudaErrors(cudaEventSynchronize(stop));
		
		float time;
		checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
		
		elapsed_time += (double)time;
		
		if(i == 0){
			checkCudaErrors(cudaMemcpy(res_val, d_val_res, M*sizeof(double), cudaMemcpyDeviceToHost));
			res->val = res_val;
			res->dim = M;
		}
	}
	

	checkCudaErrors(cudaFree(d_JA));
	checkCudaErrors(cudaFree(d_AS));
	checkCudaErrors(cudaFree(d_val_vec));
	checkCudaErrors(cudaFree(d_val_res));

	
	
	return elapsed_time;
}

unsigned long long get_device_globalmemory_size(){
	int dev = 0;
	cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    unsigned long long res = deviceProp.totalGlobalMem;
    return res;
}

long double checkDeviceGlobalMemoryOccupage(struct csrFormat *csr, struct ellpackFormat *ellpack, char *format, unsigned long long totalGlobalMem, int nz)
{
	if(strcmp(format, "-csrAdaptive") == 0){
		int *IRP = csr->IRP;
		int M = csr->M;
		int N = csr->N;

		unsigned long long size = 0;
		
		int block_count = create_row_blocks(true, M, IRP, nullptr);

		size += 1 * sizeof(int);	//M
		size += (unsigned long long)(M+1)*sizeof(int);	//IRP
		size += (unsigned long long)nz*sizeof(int);	//JA
		size += (unsigned long long)nz*sizeof(double);	//AS
		size += (unsigned long long)N*sizeof(double);	//vec->val
		size += (unsigned long long)M*sizeof(double);	//res->val
		size += (unsigned long long)block_count*sizeof(int); //block_row
		
		return (long double)size / totalGlobalMem;
	}
	if(strcmp(format, "-csrVector") == 0){
		int M = csr->M;
		int N = csr->N;
		
		unsigned long long size = 0;

		size += 1 * sizeof(int);	//M
		size += (unsigned long long)(M+1)*sizeof(int);	//IRP
		size += (unsigned long long)nz*sizeof(int);	//JA
		size += (unsigned long long)nz*sizeof(double);	//AS
		size += (unsigned long long)N*sizeof(double);	//vec->val
		size += (unsigned long long)M*sizeof(double);	//res->val
		
		
		return (long double)size / totalGlobalMem;
	}
	if(strcmp(format, "-csrcalar") == 0){
		int M = csr->M;
		int N = csr->N;
		
		unsigned long long size = 0;

		size += 1 * sizeof(int);	//M
		size += (unsigned long long)(M+1)*sizeof(int);	//IRP
		size += (unsigned long long)nz*sizeof(int);	//JA
		size += (unsigned long long)nz*sizeof(double);	//AS
		size += (unsigned long long)N*sizeof(double);	//vec->val
		size += (unsigned long long)M*sizeof(double);	//res->val

		
		return (long double)size / totalGlobalMem;
	}
	
	if(strcmp(format, "-ellpack") == 0){
		int maxnz = ellpack->MAXNZ;
		int M = ellpack->M;
		int N = ellpack->N;
		
		unsigned long long size = 0;
		
		size += 1 * sizeof(int);	//M
		size += 1 * sizeof(int);	//maxnz
		size += (unsigned long long)maxnz*M*sizeof(int);	//JA
		size += (unsigned long long)maxnz*M*sizeof(double);	//AS
		size += (unsigned long long)N*sizeof(double);	//vec->val
		size += (unsigned long long)M*sizeof(double);	//res->val

		return (long double)size / totalGlobalMem;
	}
	return -1;
}

int main(int argc, char *argv[])
{
	
	if(argc < 2 || ((strcmp(argv[1],"-csrScalar") != 0) && (strcmp(argv[1],"-csrVector") != 0) && (strcmp(argv[1],"-csrAdaptive") != 0) && (strcmp(argv[1],"-ellpack") != 0))){
		printf("Uso: './mainGPU -csrScalar' o './mainGPU -csrVector' o './mainGPU -csrAdaptive' o './mainGPU -ellpack'\n");
		exit(EXIT_FAILURE);
	}
	

	
	prepare_result_gpu_csv(argv[1]);
	
	unsigned long long deviceGlobalMemorySize = get_device_globalmemory_size();
	
	char *matrices[30];
    matrices[0] = "cage4.mtx";
    matrices[1] = "mhda416.mtx";    
    matrices[2] = "mcfe.mtx";
    matrices[3] = "olm1000.mtx";
    matrices[4] = "adder_dcop_32.mtx";
    matrices[5] = "west2021.mtx";
    matrices[6] = "cavity10.mtx";
    matrices[7] = "rdist2.mtx";
    matrices[8] = "cant.mtx";
    matrices[9] = "olafu.mtx";
    matrices[10] = "Cube_Coup_dt0.mtx";
    matrices[11] = "ML_Laplace.mtx";
    matrices[12] = "bcsstk17.mtx";
    matrices[13] = "mac_econ_fwd500.mtx";
    matrices[14] = "mhd4800a.mtx";
    matrices[15] = "cop20k_A.mtx";
    matrices[16] = "raefsky2.mtx";
    matrices[17] = "af23560.mtx";
    matrices[18] = "lung2.mtx";
    matrices[19] = "PR02R.mtx";
    matrices[20] = "FEM_3D_thermal1.mtx";
    matrices[21] = "thermal1.mtx";
    matrices[22] = "thermal2.mtx";
    matrices[23] = "thermomech_TK.mtx";
    matrices[24] = "nlpkkt80.mtx";
    matrices[25] = "webbase-1M.mtx";
    matrices[26] = "dc1.mtx";
    matrices[27] = "amazon0302.mtx";
    matrices[28] = "af_1_k101.mtx";
    matrices[29] = "roadNet-PA.mtx";
	

	
	for(int ind = 0; ind < 30; ind++){
		char *matrix_name = matrices[ind];
		char matrix_path[256];
		strcpy(matrix_path, "");
		strcat(matrix_path, MATRICES_FOLDER);
		strcat(matrix_path, matrix_name);
		
	
		printf("\nMatrix path : %s\n", matrix_path);
		fflush(stdout);
		struct sparsematrix *matrix = (struct sparsematrix *) malloc(sizeof(struct sparsematrix));
		struct vector *vec = (struct vector *) malloc(sizeof(struct vector));
		
		if(matrix == NULL){
			fprintf(stderr, "Error in memory allocation.\n");
			exit(-1);
		}
		
		printf("Lettura matrice...\n");
		int ret;
		ret = read_mm(matrix_path, matrix);

		if(ret == -2){
			printf("Matrice %s non trovata.\nSalto alla prossima matrice.\n", matrix_name);
			continue;
		}
		
		if(ret == -1){
			fprintf(stderr,"Errore nella lettura della matrice.\nFine programma.\n");
			exit(EXIT_FAILURE);
		}
		
		printf("matrice letta con successo!\n\n");
		
		
		
		printf("Creazione o lettura vettore...\n");
		
			ret = create_or_read_vector(matrix->N, vec, 0);
		if(ret != 0){
			fprintf(stderr, "Creazione o lettura vettore non riuscita. Fine programma.\n");
			exit(EXIT_FAILURE);
		}
		printf("Vettore creato o letto con successo!\n\n");
		
		struct vector *res = (struct vector *) malloc(sizeof(struct vector));
		struct vector *res_sequential = (struct vector *) malloc(sizeof(struct vector));
		int nz = matrix->nz;
		
		printf("\nInizio quicksort\n");
		fflush(stdout);
		quicksort(matrix->globalIndex, matrix->I, matrix->J, matrix->val, matrix->nz);
		

		printf("Fine quicksort\n");
		fflush(stdout);
	
		
		double elapsed_time, elapsed_time_sequential, speedup;
		double flops;
		if(strcmp(argv[1], "-csrAdaptive") == 0) {
			
			
			struct csrFormat *csr = (struct csrFormat *) malloc(sizeof(struct csrFormat));
			convert_from_matrix_to_csr(matrix, csr);
			
			if(checkDeviceGlobalMemoryOccupage(csr, nullptr, argv[1], deviceGlobalMemorySize, matrix->nz) > 1.0){
				fprintf(stdout, "La matrice %s è troppo grande per la memoria del device.\nSalto alla prossima matrice.\n", matrix_name);
				free(matrix->I);
				free(matrix->J);
				free(matrix->globalIndex);
				free(matrix->val);
				free(matrix);
				write_result_gpu_csv(argv[1], matrix_name, -1, -1, -1, -1);
				continue;
			}

			
			free(matrix->I);
			free(matrix->J);
			free(matrix->globalIndex);
			free(matrix->val);
			free(matrix);
			
			
		
		
				
			elapsed_time = csr_adaptive_cuda(csr, vec, res, nz);
				
			
		
			elapsed_time /= EXECUTIONS; //milliseconds
			flops = 2.e-6*nz/(double)elapsed_time; //giga flops
		//	fprintf(stdout,"Adaptive - Elapsed time: %lg\nGFlops: %lg\n\n", elapsed_time, flops);
		//	fflush(stdout);
			
			elapsed_time_sequential = sequential_csr(csr, vec, res_sequential);
			speedup = 100.0 * (elapsed_time_sequential / 1000.0) / elapsed_time;
			
			int correctness = check_correctness(res, res_sequential, TOLERANCE);
			if(correctness != 0){
				if(correctness == 1){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale è maggiore di %lg\nFine programma.\n", TOLERANCE);
					exit(EXIT_FAILURE);
				} else if(correctness == 2){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale non hanno la stessa dimensione.\nFine programma.\n");
					exit(EXIT_FAILURE);
				}
			}
			
			
			free(vec->val);
			free(vec);
			free(res->val);
			free(res);
			free(res_sequential->val);
			free(res_sequential);
			free(csr->AS);
			free(csr->JA);
			free(csr->IRP);
			free(csr);
		}
		
		if(strcmp(argv[1], "-csrVector") == 0) {
			struct csrFormat *csr = (struct csrFormat *) malloc(sizeof(struct csrFormat));
			convert_from_matrix_to_csr(matrix, csr);
			
			if(checkDeviceGlobalMemoryOccupage(csr, nullptr, argv[1], deviceGlobalMemorySize, matrix->nz) > 1.0){
				fprintf(stdout, "La matrice %s è troppo grande per la memoria del device.\nSalto alla prossima matrice.\n", matrix_name);
				free(matrix->I);
				free(matrix->J);
				free(matrix->globalIndex);
				free(matrix->val);
				free(matrix);
				write_result_gpu_csv(argv[1], matrix_name, -1, -1, -1, -1);
				continue;
			}
			
			free(matrix->I);
			free(matrix->J);
			free(matrix->globalIndex);
			free(matrix->val);
			free(matrix);
				
			elapsed_time = csr_vector_cuda(csr, vec, res, nz);
				
			
		
			elapsed_time /= EXECUTIONS; //milliseconds
			flops = 2.e-6*nz/(double)elapsed_time; //giga flops
	//		fprintf(stdout,"Vector - Elapsed time: %lg\nGFlops: %lg\n\n", elapsed_time, flops);
	//		fflush(stdout);
			
			elapsed_time_sequential = sequential_csr(csr, vec, res_sequential);
			speedup = 100.0 * (elapsed_time_sequential / 1000.0) / elapsed_time;
			
			int correctness = check_correctness(res, res_sequential, TOLERANCE);
			if(correctness != 0){
				if(correctness == 1){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale è maggiore di %lg\nFine programma.\n", TOLERANCE);
					exit(EXIT_FAILURE);
				} else if(correctness == 2){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale non hanno la stessa dimensione.\nFine programma.\n");
					exit(EXIT_FAILURE);
				}
			}
			
			
			free(vec->val);
			free(vec);
			free(res->val);
			free(res);
			free(res_sequential->val);
			free(res_sequential);
			free(csr->AS);
			free(csr->JA);
			free(csr->IRP);
			free(csr);
		}
		
		if(strcmp(argv[1], "-csrScalar") == 0) {
			struct csrFormat *csr = (struct csrFormat *) malloc(sizeof(struct csrFormat));
			convert_from_matrix_to_csr(matrix, csr);
			
			if(checkDeviceGlobalMemoryOccupage(csr, nullptr, argv[1], deviceGlobalMemorySize, matrix->nz) > 1.0){
				fprintf(stdout, "La matrice %s è troppo grande per la memoria del device.\nSalto alla prossima matrice.\n", matrix_name);
				free(matrix->I);
				free(matrix->J);
				free(matrix->globalIndex);
				free(matrix->val);
				free(matrix);
				write_result_gpu_csv(argv[1], matrix_name, -1, -1, -1, -1);
				continue;
			}
			
			free(matrix->I);
			free(matrix->J);
			free(matrix->globalIndex);
			free(matrix->val);
			free(matrix);

				
			elapsed_time = csr_scalar_cuda(csr, vec, res, nz);
				
			
		
			elapsed_time /= EXECUTIONS; //milliseconds
			flops = 2.e-6*nz/(double)elapsed_time; //giga flops
//			fprintf(stdout,"Scalar - Elapsed time: %lg\nGFlops: %lg\n\n", elapsed_time, flops);
	//		fflush(stdout);
			
			
			elapsed_time_sequential = sequential_csr(csr, vec, res_sequential);
			speedup = 100.0 * (elapsed_time_sequential / 1000.0) / elapsed_time;
			
			int correctness = check_correctness(res, res_sequential, TOLERANCE);
			if(correctness != 0){
				if(correctness == 1){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale è maggiore di %lg\nFine programma.\n", TOLERANCE);
					exit(EXIT_FAILURE);
				} else if(correctness == 2){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale non hanno la stessa dimensione.\nFine programma.\n");
					exit(EXIT_FAILURE);
				}
			}
			
			
			free(vec->val);
			free(vec);
			free(res->val);
			free(res);
			free(res_sequential->val);
			free(res_sequential);
			free(csr->AS);
			free(csr->JA);
			free(csr->IRP);
			free(csr);
		}
	 
		if(strcmp(argv[1], "-ellpack") == 0) {
			struct ellpackFormat *ellpack = (struct ellpackFormat *) malloc(sizeof(struct ellpackFormat));
			if(convert_from_matrix_to_ellpack(matrix, ellpack) == -1){
				fprintf(stderr, "La matrice %s è troppo grande per essere memorizzata in memoria.\nSalto alla prossima matrice.\n", matrix_name);
				free(matrix->I);
				free(matrix->J);
				free(matrix->globalIndex);
				free(matrix->val);
				free(matrix);
				write_result_gpu_csv(argv[1], matrix_name, -1, -1, -1, -1);
				continue;
			}
			
			if(checkDeviceGlobalMemoryOccupage(nullptr, ellpack, argv[1], deviceGlobalMemorySize, matrix->nz) > 1.0){
				fprintf(stdout, "La matrice %s è troppo grande per la memoria del device.\nSalto alla prossima matrice.\n", matrix_name);
				free(matrix->I);
				free(matrix->J);
				free(matrix->globalIndex);
				free(matrix->val);
				free(matrix);
				write_result_gpu_csv(argv[1], matrix_name, -1, -1, -1, -1);
				continue;
			}
			
			free(matrix->I);
			free(matrix->J);
			free(matrix->globalIndex);
			free(matrix->val);
			free(matrix);
			
			//nota: deve essere fatto prima della trasposizione
			elapsed_time_sequential = sequential_ellpack(ellpack, vec, res_sequential);
			
			
			
			traspose_ellpack(ellpack);
			
				
				   
			elapsed_time = ellpack_cuda(ellpack, vec, res, nz);
						
				
			
			elapsed_time /= EXECUTIONS; //milliseconds

			flops = 2.e-6*nz/(double)elapsed_time; //giga flops
	
			
			speedup = 100.0 * (elapsed_time_sequential / 1000.0) / elapsed_time;
			
			int correctness = check_correctness(res, res_sequential, TOLERANCE);
			if(correctness != 0){
				if(correctness == 1){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale è maggiore di %lg\nFine programma.\n", TOLERANCE);
					exit(EXIT_FAILURE);
				} else if(correctness == 2){
					fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale non hanno la stessa dimensione.\nFine programma.\n");
					exit(EXIT_FAILURE);
				}
			}
			
			
	//		fprintf(stdout,"ELLPACK - Elapsed time: %lg\nGFlops: %lg\n\n", elapsed_time, flops);
	//		fflush(stdout);
				
		
			free(vec->val);
			free(vec);
			free(res->val);
			free(res);
			free(res_sequential->val);
			free(res_sequential);
			free(ellpack->AS);
			free(ellpack->JA);
			free(ellpack);
			
		}
		write_result_gpu_csv(argv[1], matrix_name, elapsed_time, flops, speedup, elapsed_time_sequential/1000);
		

	}
	

    printf("Fine programma!\n");
    exit(EXIT_SUCCESS);
	
	
}

