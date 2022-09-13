#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include "./IO/mmio.h"
#include "./IO/io.h"
#include "./IO/utils.h"
#include <omp.h>
#include "compute.h"
#include <unistd.h>
#define MAX_NUM_THREADS 40
#define EXECUTIONS 32
#define MATRICES_FOLDER "/data/matrici/"
#define TOLERANCE 10E-7

int main(int argc, char *argv[])
{
	
	if(argc < 2 || ((strcmp(argv[1],"-csr") != 0) &&  (strcmp(argv[1],"-ellpack") != 0))){
		printf("Uso: './mainCPU -csr' o './mainCPU -ellpack'\n");
		exit(EXIT_FAILURE);
	}

	prepare_result_cpu_csv(argv[1]);

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
			printf("Matrice %s non trovata.\nSalto alla prossima matrice.\n", matrix_path);
			continue;
		}
		
		if(ret == -1){
			fprintf(stderr,"Errore nella lettura della matrice.\nFine programma.\n");
			exit(EXIT_FAILURE);
		}
		printf("matrice letta con successo!\n\n");
		/************************/
		/* now write out matrix */
		/************************/

	  //  mm_write_banner(stdout, matcode);
	  /*
		mm_write_mtx_crd_size(stdout, matrix->M, matrix->N, matrix->nz);
		for (int i=0; i<matrix->nz; i++)
			fprintf(stdout, "%d %d %20.19g\n", matrix->I[i], matrix->J[i], matrix->val[i]);
		fprintf(stdout, "\n\n");
	*/
		
		printf("Creazione o lettura vettore...\n");
		ret = create_or_read_vector(matrix->N, vec, 0);
		if(ret != 0){
			fprintf(stderr, "Creazione o lettura vettore non riuscita. Fine programma.\n");
			exit(EXIT_FAILURE);
		}
		printf("Vettore creato o letto con successo!\n\n");
		
		struct vector *res = (struct vector *) malloc(sizeof(struct vector));

		int nz = matrix->nz;
		
		//sequentialMatVecMult(matrix, vec, res);
		/*
		for(int r = 0; r < matrix->M; r++){
			printf("%d %20.19g\n", r, res->val[r]); 
			
		}
		* */
		//writeVecOnFile(res);
		
		
	
		printf("Inizio quicksort\n");
		fflush(stdout);
		quicksort(matrix->globalIndex, matrix->I, matrix->J, matrix->val, matrix->nz);
		

		printf("Fine quicksort\n\n");
		fflush(stdout);
		
		
		double elapsed_time, elapsed_time_sequential, speedup;
		double cpu_flops;
		double elapsed_time_list[MAX_NUM_THREADS], speedup_list[MAX_NUM_THREADS];
		double cpu_flops_list[MAX_NUM_THREADS];
		if(strcmp(argv[1], "-csr") == 0) {
			struct csrFormat *csr = (struct csrFormat *) malloc(sizeof(struct csrFormat));
			convert_from_matrix_to_csr(matrix, csr);
			free(matrix->I);
			free(matrix->J);
			free(matrix->globalIndex);
			free(matrix->val);
			free(matrix);
			
			struct vector *res_sequential = (struct vector *) malloc(sizeof(struct vector));
			elapsed_time_sequential = sequential_csr(csr, vec, res_sequential);
			
			double *res_val = (double *) calloc (csr->M, sizeof(double)); 
			res->val = res_val;
			for(int i = 1; i <= MAX_NUM_THREADS; i++){
				omp_set_num_threads(i);
				elapsed_time = 0.0;
				
				for(int j = 0; j < EXECUTIONS; j++){
						
					elapsed_time += csr_simd_reduction(csr, vec, res, nz);;
		
				}
			
				elapsed_time /= (1000.0 * EXECUTIONS); //milliseconds
				elapsed_time_list[i-1] = elapsed_time;
				cpu_flops = 2.e-6*nz/(double)elapsed_time; //giga flops
				cpu_flops_list[i-1] = cpu_flops;
			//	fprintf(stdout,"Num threads: %d\nElapsed time: %lg\nGFlops: %lg\n\n", i, elapsed_time, cpu_flops);
			//	fflush(stdout);
			
				
				
				speedup = 100.0 * (elapsed_time_sequential / 1000.0) / elapsed_time;

				speedup_list[i-1] = speedup;
				int correctness = check_correctness(res, res_sequential, TOLERANCE);
				if(correctness != 0){
					if(correctness == 1){
						fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale è maggiore di %lg\nFine programma.\n", TOLERANCE);
						exit(-1);
					} else if(correctness == 2){
						fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale non hanno la stessa dimensione.\nFine programma.\n");
						exit(-1);
					}
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
				double temp_list[MAX_NUM_THREADS];
				for(int temp = 0; temp < MAX_NUM_THREADS; temp ++)
					temp_list[temp] = -1;
				write_result_cpu_csv(argv[1], matrix_name, MAX_NUM_THREADS, temp_list, temp_list, temp_list, -1);
				continue;
			}
			
			free(matrix->I);
			free(matrix->J);
			free(matrix->globalIndex);
			free(matrix->val);
			free(matrix);
			
			struct vector *res_sequential = (struct vector *) malloc(sizeof(struct vector));
			elapsed_time_sequential = sequential_ellpack(ellpack, vec, res_sequential);
			
			double *res_val = (double *) calloc (ellpack->M, sizeof(double)); //inizializza a 0
			res->val = res_val;
			
			for(int i = 1; i <= MAX_NUM_THREADS; i++){
				omp_set_num_threads(i);
				elapsed_time = 0.0;
				for(int j = 0; j < EXECUTIONS; j++){
				   
					elapsed_time += ellpack_simd_reduction(ellpack, vec, res, nz);
					
				}
			
				elapsed_time /= (1000.0 * EXECUTIONS); //milliseconds
				elapsed_time_list[i-1] = elapsed_time;
				cpu_flops = 2.e-6*nz/(double)elapsed_time; //giga flops
				cpu_flops_list[i-1] = cpu_flops;
				
			//	fprintf(stdout,"Num threads: %d\nElapsed time: %lg\nGFlops: %lg\n\n", i, elapsed_time, cpu_flops);
			//	fflush(stdout);
				
				
			

				speedup = 100.0 * (elapsed_time_sequential / 1000.0) / elapsed_time;
				speedup_list[i-1] = speedup;
				int correctness = check_correctness(res, res_sequential, TOLERANCE);
				if(correctness != 0){
					if(correctness == 1){
						fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale è maggiore di %lg\nFine programma.\n", TOLERANCE);
						exit(-1);
					} else if(correctness == 2){
						fprintf(stderr, "La differenza tra il risultato del calcolo parallelo e quello seriale non hanno la stessa dimensione.\nFine programma.\n");
						exit(-1);
					}
				}
				
			}
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
		write_result_cpu_csv(argv[1], matrix_name, MAX_NUM_THREADS, elapsed_time_list, cpu_flops_list, speedup_list, elapsed_time_sequential/1000.0);
		
		
		
		
		
	}

    printf("Fine programma!\n");
    exit(EXIT_SUCCESS);
	
}
