#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <string.h>

#include "mmio.h"

#include "io.h"
#include "utils.h"

#define RESULT_CPU_CSR "resultCPUCsr.csv"
#define RESULT_CPU_ELLPACK "resultCPUEllpack.csv"
#define RESULT_GPU_CSR "resultGPU"
#define RESULT_GPU_ELLPACK "resultGPUEllpack.csv"

int create_or_read_vector(int dim, struct vector *vec, int forceCreation)
{ 	
	extern int errno;
	
	int lenFilename = get_vector_filename_len(dim);
	char filename[lenFilename];

	sprintf(filename, "./vettori/%d.txt",dim);
	
	FILE *f;
	
	double *val;
	
	val = (double *) malloc (dim * sizeof(double));

	if(val == NULL){
		perror("Errore in malloc(): ");
		exit(-1);
	}
	if(forceCreation == 1)
	{
		fill_double_array(dim, val);
		f = fopen(filename, "w");
		for(int i = 0; i < dim; i++){
			fprintf(f, "%lg\n", val[i]);
		}
		fflush(f);
		if(f != stdout)
			fclose(f);
	} else {
		
		f = fopen(filename, "r");
		if(f == NULL && errno != ENOENT)  
		{
			fprintf(stderr, "Could not open vector file.\n");
			return -1;
		}
		else if (f == NULL && errno == ENOENT) // file doesn't exists -> creation
		{
			printf("\t\t- creazione file vettore\n");
			fill_double_array(dim, val);
			fflush(stdout);
			f = fopen(filename, "w");

			fflush(stdout);
			if(f == NULL)
				perror("Errore: ");
			for(int i = 0; i < dim; i++){
				fflush(stdout);
				fprintf(f, "%lf\n", val[i]);
			}
			fflush(stdout);
			fflush(f);
			if(f != stdout)
				fclose(f);
			
		} else 
		{ //file exists and is readable
			printf("\t\t- lettura file vettore\n");
			for(int i = 0; i < dim; i++){
				if(fscanf(f, "%lg\n", &val[i]) != 1){
					fprintf(stderr, "Errore nella lettura del vettore.\nFine programma.\n");
					exit(-1);
				}
			}
			
			if(f != stdin)
				fclose(f);
		}
	}
	
	vec->dim = dim;
	vec->val = val;
	return 0;
}

int read_mm(char *filename, struct sparsematrix *matrix)
{
	extern int errno;
	
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    unsigned long long  *globalIndex;
    double *val;
	printf("%s\n", filename);
	printf("\t\t- apertura file\n");
	if ((f = fopen(filename, "r")) == NULL) {
		if(errno == ENOENT){
			return -2;
		} else {
			perror("Error: ");
			return -1;
		}
	
	}
    
	printf("\t\t- lettura banner\n");
    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        return -1;
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
	
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        return -1;

	printf("\t\t- allocazione memoria\n");
    /* reseve memory for matrix*/
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    globalIndex = (unsigned long long  *) malloc(nz * sizeof(unsigned long long ));
    val = (double *) malloc(nz * sizeof(double));
    if(I == NULL || J == NULL || val == NULL){
		fprintf(stderr, "Error in memory allocation\n");
		return -1;
	}
	

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	/* NOTE: for "pattern" matrix the values are not provided since they */
	/*   are assumed to be 1.0 */
	printf("\t\t- lettura matrice\n");
	if(mm_is_pattern(matcode)){
		for (i=0; i<nz; i++)
		{	
			if(fscanf(f, "%d %d\n", &I[i], &J[i]) != 2){
					printf("%d %d\n", I[i], J[i]);
					fprintf(stderr, "Errore nella lettura della matrice.\nFine programma.\n");
					exit(-1);
				}
			val[i] = 1.0;
			I[i]--;  /* adjust from 1-based to 0-based */
			J[i]--;
			globalIndex[i] = (unsigned long long)I[i] * (unsigned long long)N + (unsigned long long)J[i];
		}
	} else {
		for (i=0; i<nz; i++){
			if(fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3){
				fprintf(stderr, "Errore nella lettura della matrice.\nFine programma.\n");
				exit(-1);
			}
			I[i]--;  /* adjust from 1-based to 0-based */
			J[i]--;
			globalIndex[i] = (unsigned long long)I[i] * (unsigned long long)N + (unsigned long long)J[i];
		}
		
	}

	/* NOTE: if the matrix is symmetrical, it is a square matrix with      */ 
	/*   a_ij = a_ji, and only the elements on or above the main diagonal  */ 
	/*   are provided, the others are known by symmetry, but we need to    */
	/*   reconstruct those elements.*/
	int actual_nz;
	int *actual_I, *actual_J;
	unsigned long long *actual_globalIndex;
	double *actual_val;

	if(mm_is_symmetric(matcode)){
		printf("\t\t- matrice simmetrica, costruzione della seconda metà\n");
		/* Count the actual non-zeros: if an element is not on the main */
		/*   diagonal, we need to count it twice, since there is its    */ 
		/*   simmetrical. */
		
		actual_nz = nz;
		for(int i = 0; i < nz; i++)
		{
			if(I[i] != J[i]) //we have found a non-zero outside the main diagonal
			{
				actual_nz ++;
			}
		}
		printf("\t\t\t- allocazione memoria per l'intera matrice\n");
		/* Memory allocation for the entire matrix */
		actual_I = (int *) malloc(actual_nz * sizeof(int));
		actual_J = (int *) malloc(actual_nz * sizeof(int));
		actual_globalIndex = (unsigned long long *) malloc(actual_nz * sizeof(unsigned long long ));
		actual_val = (double *) malloc(actual_nz * sizeof(double));
		
		if(actual_I == NULL || actual_J == NULL || actual_val == NULL){
			fprintf(stderr, "Error in memory allocation\n");
			return -1;
		}
		printf("\t\t\t- ricostruzione della matrice\n");
		/* when we find a value outside the main diagonal, we put its   
		 *     symmetrical at the end of the list.*/
		int backwardIndex = 0; 
		for(int j = 0; j < nz; j++)
		{
			actual_I[j] = I[j];
			actual_J[j] = J[j];
			actual_globalIndex[j] = (unsigned long long)actual_I[j] * (unsigned long long)N + (unsigned long long)actual_J[j];
			actual_val[j] = val[j];
			
			
			/* TODO: check for error */
			if(I[j] != J[j])
			{
				actual_I[actual_nz - 1 - backwardIndex] = J[j]; 
				actual_J[actual_nz - 1 - backwardIndex] = I[j];
				actual_globalIndex[actual_nz - 1 - backwardIndex] = (unsigned long long)actual_I[actual_nz - 1 - backwardIndex] * (unsigned long long)N + (unsigned long long)actual_J[actual_nz - 1 - backwardIndex];
				actual_val[actual_nz - 1 - backwardIndex] = val[j]; 
				backwardIndex ++;
			}
			
		}
		
		free(I);
		free(J);
		free(globalIndex);
		free(val);
		
		nz = actual_nz; 
		I = actual_I;
		J = actual_J;
		globalIndex = actual_globalIndex;
		val = actual_val;
		
	
	}

    if (f !=stdin) 
		fclose(f);
	
	matrix->M = M;
	matrix->N = N;
	matrix->nz = nz;
	matrix->I = I;
	matrix->J = J;
	matrix->globalIndex = globalIndex;
	matrix->val = val;
	
	
    

	return 0;
	
	
	
}

int prepare_result_gpu_csv(char *format)
{
	FILE *f;
	if(strcmp(format, "-csrScalar") == 0){
		char filename[64];
		strcpy(filename, "");
		strcat(filename, RESULT_GPU_CSR);
		strcat(filename, "csrScalar.csv");
		f = fopen(filename, "w");
		if (f == NULL) 
			return -1; 
	}
	if(strcmp(format, "-csrVector") == 0){
		char filename[64];
		strcpy(filename, "");
		strcat(filename, RESULT_GPU_CSR);
		strcat(filename, "csrVector.csv");
		f = fopen(filename, "w");
		if (f == NULL) 
			return -1; 
	} 
	if(strcmp(format, "-csrAdaptive") == 0){
		char filename[64];
		strcpy(filename, "");
		strcat(filename, RESULT_GPU_CSR);
		strcat(filename, "csrAdaptive.csv");
		f = fopen(filename, "w");
		if (f == NULL) 
			return -1;  
	} 
	if(strcmp(format, "-ellpack") == 0){
		f = fopen(RESULT_GPU_ELLPACK, "w");
		if (f == NULL) 
			return -1; 
	}   
		
	if(fprintf(f, "matrix,elapsed_time (millis),flops (GF),speedup,elapsed_time_sequential (millis)\n") < 0)
		return -1;

	fflush(f);
	
	fclose(f);
	return 0;
}

int write_result_gpu_csv(char *format, char *matrix_name, double time, double flops, double speedup, double elapsed_time_sequential)
{
	FILE *f;
	if(strcmp(format, "-csrScalar") == 0){
		char filename[64];
		strcpy(filename, "");
		strcat(filename, RESULT_GPU_CSR);
		strcat(filename, "csrScalar.csv");
		if ((f = fopen(filename, "a")) == NULL) 
			return -1; 
	}
	if(strcmp(format, "-csrVector") == 0){
		char filename[64];
		strcpy(filename, "");
		strcat(filename, RESULT_GPU_CSR);
		strcat(filename, "csrVector.csv");
		if ((f = fopen(filename, "a")) == NULL) 
			return -1; 
	} 
	if(strcmp(format, "-csrAdaptive") == 0){
		char filename[64];
		strcpy(filename, "");
		strcat(filename, RESULT_GPU_CSR);
		strcat(filename, "csrAdaptive.csv");
		if ((f = fopen(filename, "a")) == NULL) 
			return -1; 
	} 
	if(strcmp(format, "-ellpack") == 0){
		if ((f = fopen(RESULT_GPU_ELLPACK, "a")) == NULL) 
			return -1; 
	}  
	if(time == -1){
		fprintf(f, "%s,,,,\n", matrix_name);
	} else {
		if(fprintf(f, "%s,%lg,%lg,%lg,%lg\n", matrix_name, time, flops, speedup, elapsed_time_sequential) < 0)
			return -1;
	}
	fflush(f);
	
	fclose(f);
	return 0;
}

int prepare_result_cpu_csv(char *format)
{
	FILE *f;
	if(strcmp(format, "-csr") == 0){
		if ((f = fopen(RESULT_CPU_CSR, "w")) == NULL) 
			return -1; 
	} 
	if(strcmp(format, "-ellpack") == 0){
		if ((f = fopen(RESULT_CPU_ELLPACK, "w")) == NULL) 
			return -1; 
	}  
		
	if(fprintf(f, "matrix,num_threads,elapsed_time (millis),flops (GF),speedup,elapsed_time_sequential (millis)\n") < 0)
		return -1;

	fflush(f);
	
	fclose(f);
	return 0;
}

int write_result_cpu_csv(char *format, char *matrix_name, int max_thread_num, double *time, double *cpu_flops, double *speedup, double elapsed_time_sequential)
{
	FILE *f;
	if(strcmp(format, "-csr") == 0){
		if ((f = fopen(RESULT_CPU_CSR, "a")) == NULL) 
			return -1; 
	} 
	if(strcmp(format, "-ellpack") == 0){
		if ((f = fopen(RESULT_CPU_ELLPACK, "a")) == NULL) 
			return -1; 
	} 
	for(int i = 1; i <= max_thread_num; i++){
		if(elapsed_time_sequential == -1){
			fprintf(f, "%s,%d,,,,\n", matrix_name, i);
		} else {
			if(fprintf(f, "%s,%d,%lg,%lg,%lg,%lg\n", matrix_name, i, time[i-1], cpu_flops[i-1], speedup[i-1], elapsed_time_sequential) < 0)
				return -1;
		}
	}
	fflush(f);
	
	fclose(f);
	return 0;
}
