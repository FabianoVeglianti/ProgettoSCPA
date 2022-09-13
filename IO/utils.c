#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "io.h"
#include "utils.h"
#include <errno.h>
#include <sys/sysinfo.h>

long double absolute_value(long double x){
	if(x >= 0)
		return x;
	else
		return -x;
}

int check_correctness(struct vector *vec1, struct vector *vec2, long double tolerance){
	int dim1 = vec1->dim;
	int dim2 = vec2->dim;
	double *val_vec1 = vec1->val;
	double *val_vec2 = vec2->val;
	
	if(dim1 != dim2)
		return 2;
	long double distance = 0.0;
	for(int i = 0; i < dim1; i++){
		if(val_vec1[i] != 0 && val_vec2[i] == 0)
			distance += (long double)absolute_value((long double)val_vec1[i] - (long double)val_vec2[i])/absolute_value((long double)val_vec1[i]);
		else if(val_vec2[i] != 0 && val_vec1[i] == 0)
			distance += (long double)absolute_value((long double)val_vec1[i] - (long double)val_vec2[i])/absolute_value((long double)val_vec2[i]);
		else if(val_vec2[i] == 0 && val_vec1[i] == 0)
			distance += 0;
		else
			distance += (long double)absolute_value((long double)val_vec1[i] - (long double)val_vec2[i])/absolute_value((long double)val_vec1[i]);
	}

	if(distance < tolerance)
		return 0;
	else
		return 1;
}

int get_vector_filename_len(int dim)
{
	int digitInDim = 1;
	double divider = 10.0;
	double res = (double) dim / divider;
	while (0)
	{
		if(res <= 1.0)
			break;
		digitInDim +=1;
		divider *= 10;
		res = (double) dim / divider;
	}
	
	int lenFilename = strlen("./matrices/vectors/.txt")+digitInDim+1;
	return lenFilename;
}

double get_random_double(){
	double range = (double) MAX - (double) MIN;
	double randomIn01 = rand()/(double)RAND_MAX;
	return (double) MIN + randomIn01 * range;
}

void fill_double_array(int len, double *array){
	for(int i = 0; i < len; i++){
		array[i] = get_random_double();
	}	
}


void quicksort(unsigned long long *globalIndex, int *I, int *J, double *val, int n)
{
	int i, j, m;

	double t;
	unsigned long long p, s;
	if (n < 2)
		return;
	p = globalIndex[n / 2];

	for (i = 0, j = n - 1;; i++, j--) {
		while (globalIndex[i]<p)
			i++;
		while (p<globalIndex[j])
			j--;
		if (i >= j)
			break;
		t = val[i];
		val[i] = val[j];
		val[j] = t;

		s = globalIndex[i];
		globalIndex[i] = globalIndex[j];
		globalIndex[j] = s;

		m = I[i];
		I[i] = I[j];
		I[j] = m;

		m = J[i];
		J[i] = J[j];
		J[j] = m;
	}
	quicksort(globalIndex, I, J, val, i);
	quicksort(globalIndex + i, I + i, J + i, val + i, n - i);
}


void convert_from_matrix_to_csr(struct sparsematrix *matrix, struct csrFormat *csr)
{
	csr->M = matrix->M;
	csr->N = matrix->N;
	
	//quicksortSparseMatrix(matrix, 0, matrix->nz-1);
	
	double *V = (double *) calloc(matrix->nz, sizeof(double));
	int *col_index = (int *) calloc(matrix->nz, sizeof(int));
	int *row_index = (int *) calloc((matrix->M+1), sizeof(int));
	
	int *counters = (int *) calloc(matrix->M, sizeof(int));
	for (int i=0; i<matrix->nz; i++)
		counters[matrix->I[i]] += 1;

	
	
	row_index[0] = 0;
	
	for(int i = 0; i < matrix->M; i++){
		row_index[i + 1] = row_index[i] + counters[i];

		
	}

	free(counters);
	
	for (int i=0; i<matrix->nz; i++){
		V[i] = matrix->val[i];
		col_index[i] = matrix->J[i];
	}
	
	
	
	csr->IRP = row_index;
	csr->JA = col_index;
	csr->AS = V;
	
	
}

unsigned long get_free_ram(){	
	 struct sysinfo info;
	sysinfo(&info);
	return info.freeram;
}

int  convert_from_matrix_to_ellpack(struct sparsematrix *matrix, struct ellpackFormat *ellpack)
{
	extern int errno;
	ellpack->M = matrix->M;
	ellpack->N = matrix->N;

	//quicksortSparseMatrix(matrix, 0, matrix->nz-1);
	int *counters = (int *) calloc(matrix->M, sizeof(int));
	for (int i=0; i<matrix->nz; i++)
		counters[matrix->I[i]] += 1;

	int maxnz = 0;
	for (int j = 0; j < matrix->M; j++){
		if(counters[j] > maxnz)
			maxnz = counters[j];
	}
	
	ellpack->MAXNZ = maxnz;
	free(counters);


	unsigned long freeRam = get_free_ram();
	if((unsigned long long)matrix->M * maxnz * (sizeof(int) + sizeof(double)) > freeRam){
		return -1; //controllo non sufficiente, la matrice potrebbe essere memorizzata, ma i vettori vec e res potrebbero non entrare in memoria.
	}

	int *JA = (int *) malloc(matrix->M * maxnz * sizeof(int));
	if(JA == NULL){
		fprintf(stderr, "Errore in allocazione memoria JA\n");
		perror("Errore: ");
		exit(EXIT_FAILURE);
	}

	double *AS = (double *) malloc(matrix->M * maxnz * sizeof(double));
	if(AS == NULL){
		fprintf(stderr, "Errore in allocazione memoria AS\n");
		perror("Errore: ");
		exit(EXIT_FAILURE);
	}
	
	

	unsigned long long sparse_matrix_element_index = 0;
	for (unsigned long long i = 0; i < ellpack->M; i++){
		int last_col_index_in_row = 0;
		for(unsigned long long j = 0; j < maxnz; j++){
			if(sparse_matrix_element_index < matrix->nz){
				if(matrix->I[sparse_matrix_element_index] == i){
					JA[i * maxnz + j] = matrix->J[sparse_matrix_element_index];
					AS[i * maxnz + j] = matrix->val[sparse_matrix_element_index];
					last_col_index_in_row = matrix->J[sparse_matrix_element_index];
					sparse_matrix_element_index += 1;
				} else {
					JA[i * maxnz + j] = last_col_index_in_row;
					AS[i * maxnz + j] = 0;
				}
			} else {
				JA[i * maxnz + j] = last_col_index_in_row;
				AS[i * maxnz + j] = 0;
			}
		}
	}
	
	ellpack->JA = JA;
	ellpack->AS = AS;
	
	return 0;
}

void traspose_ellpack(struct ellpackFormat *ellpack){
	int M = ellpack->M;
	int maxnz = ellpack->MAXNZ;
	int *JA_t = (int *) calloc(M * maxnz, sizeof(int));
	double *AS_t = (double *) calloc(M * maxnz, sizeof(double));
	
	for(int i = 0; i < M; i++){
		for(int j = 0; j < maxnz; j++){
			JA_t[j * M + i] = ellpack->JA[i * maxnz + j];
			AS_t[j * M + i] = ellpack->AS[i * maxnz + j];
		}
	}
	
	free(ellpack->JA);
	free(ellpack->AS);
	
	ellpack->JA = JA_t;
	ellpack->AS = AS_t;
	
}
