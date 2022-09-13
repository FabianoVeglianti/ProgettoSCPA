/*
 * M, N and nz: number of rows, number of columns and number of non-zeros
 * I and J: row indices array and col indices array
 * val: values array 
 * */
struct sparsematrix {
	int M, N, nz;   
    int *I, *J;
    double *val;
    unsigned long long *globalIndex;
};

struct vector {
	int dim;
	double *val;
};

int read_mm(char *filename, struct sparsematrix *matrix);
int create_or_read_vector(int dim, struct vector *vec, int forceCreation);

int prepare_result_cpu_csv(char *format);
int write_result_cpu_csv(char *format, char *matrix_name, int max_thread_num, double *time, double *cpu_flops, double *speedup, double time_sequential);

int prepare_result_gpu_csv(char *format);
int write_result_gpu_csv(char *format, char *matrix_name, double time, double flops, double speedup, double elapsed_time_sequential);
