#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "./IO/io.h"
#include "./IO/utils.h"

#define NZ_PER_CHUNK 10000

double sequential_csr(struct csrFormat *csr, struct vector *vec, struct vector *res)
{
	int M = csr->M;
	double *val = (double *) calloc (M, sizeof(double)); //inizializza a 0
	struct timeval start,end;
	
	gettimeofday(&start, NULL);
	double result;
	int row_start_index, row_end_index;
	for(int row = 0; row < M; row++){
		row_start_index = csr->IRP[row]; 
		row_end_index = csr->IRP[row+1];
		result = 0.0;
		for(int j = row_start_index; j < row_end_index; j++){
			int col_index = csr->JA[j];
			double mval = csr->AS[j];
			double vval = vec->val[col_index];
			result += mval * vval; 
		}
		val[row] = result;
	}
	
	gettimeofday(&end, NULL);
	
	res->dim = M;
	res->val = val;
	
	long t = (end.tv_sec - start.tv_sec)*1000000.0 + end.tv_usec - start.tv_usec;

	return (double) t;
} 

double sequential_ellpack(struct ellpackFormat *ellpack, struct vector *vec, struct vector *res)
{
	int M = ellpack->M;
	int maxnz = ellpack->MAXNZ;
	double *val = (double *) calloc (M, sizeof(double)); //inizializza a 0
	struct timeval start,end;
	
	gettimeofday(&start, NULL);
	double result;
	int row;
	for(int i = 0; i < M; i++){
		row = i * maxnz;
		result = 0.0;
		for(int j = 0; j < maxnz; j++){
			int col_index = ellpack->JA[row+j];
			double mval = ellpack->AS[row+j];
			double vval = vec->val[col_index];
			result += mval * vval; 
		}
		val[i] = result;
	}
	
	gettimeofday(&end, NULL);
	
	res->dim = M;
	res->val = val;
	
	long t = (end.tv_sec - start.tv_sec)*1000000.0 + end.tv_usec - start.tv_usec;


	return (double) t;
} 


double csr_simd_reduction(struct csrFormat *csr, struct vector *vec, struct vector *res, int nz){
	int M = csr->M;
	int nonzerosPerRow = nz/M;
	int chunk_size = NZ_PER_CHUNK/nonzerosPerRow;
	
	double *val = res->val;
	
	struct timeval start,end;
	
	gettimeofday(&start, NULL);
	
	
	
	
	int i,j,tmp;
	#pragma omp parallel 
	{
		#pragma omp for private(i, j, tmp) schedule(dynamic, chunk_size)
		for (i=0; i<M; i++)
		{
			double result = 0.0;
			#pragma omp simd reduction(+ : result)
			for (j = csr->IRP[i]; j < csr->IRP[i+1]; j++)
			{
				tmp = csr->JA[j];
				result += csr->AS[j] * vec->val[tmp];
			}
			val[i] = result;
		}
	}
	
	gettimeofday(&end, NULL);
	
	res->dim = M;
	
	
	//printf("%ld.%06ld\n", start.tv_sec, start.tv_usec);
	//printf("%ld.%06ld\n", end.tv_sec, end.tv_usec);
	long t = (end.tv_sec - start.tv_sec)*1000000.0 + end.tv_usec - start.tv_usec;

	return (double) t;
}


double ellpack_simd_reduction(struct ellpackFormat *ellpack, struct vector *vec, struct vector *res, int nz){
	int M = ellpack->M;
	int maxnz = ellpack->MAXNZ;

	
	
	
	double *val = res->val;
	struct timeval start,end;
	
	gettimeofday(&start, NULL);
	
	
	
	
	int i,j,tmp, row;
	#pragma omp parallel 
	{
		#pragma omp for private(i, j, tmp, row) schedule(static) 
		for (i=0; i<M; i++)
		{
			double result = 0.0;
			row = i * maxnz;
			#pragma omp simd reduction(+ : result)
			for (j = 0; j < maxnz; j++)
			{
				tmp = ellpack->JA[row + j];
				result += ellpack->AS[row + j] * vec->val[tmp];
			}
			val[i] = result;
		}
	}

	gettimeofday(&end, NULL);
	
	res->dim = M;

	//printf("%ld.%06ld\n", start.tv_sec, start.tv_usec);
	//printf("%ld.%06ld\n", end.tv_sec, end.tv_usec);
	long t = (end.tv_sec - start.tv_sec)*1000000.0 + end.tv_usec - start.tv_usec;

	
	
	return (double) t;
}
