#define MAX 10.0
#define MIN -10.0


struct csrFormat {
	int M, N;
	int *IRP;
	int *JA;
	double *AS;
};

struct ellpackFormat {
	int M, N;
	int MAXNZ;
	int *JA;
	double *AS;
};


int get_vector_filename_len(int dim);
void fill_double_array(int len, double *array);
void quicksort(unsigned long long *globalIndex, int *I, int *J, double *val, int n);


void convert_from_matrix_to_csr(struct sparsematrix *matrix, struct csrFormat *csr);

int convert_from_matrix_to_ellpack(struct sparsematrix *matrix, struct ellpackFormat *ellpack);

void traspose_ellpack(struct ellpackFormat *ellpack);

int check_correctness(struct vector *vec1, struct vector *vec2, long double tolerance);
