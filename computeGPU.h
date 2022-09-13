__global__ void csr_scalar(int n_rows, const int *JA, const int *IRP, const double *AS, const double *val_vec, double *val_res);

__global__ void csr_vector(int n_rows, const int *JA, const int *IRP, const double *AS,	const double *val_vec, double *val_res);
__device__ double warp_reduce(double value);

int create_row_blocks(bool onlyCount,int n_rows, const int *IRP, int *row_blocks);
__device__ int prev_power_of_2(int n);
__global__ void csr_adaptive(int n_rows, const int *JA, const int *IRP, const double *AS, const double *val_vec, double *val_res, const int *row_blocks);

__global__ void ellpack_kernel(int n_rows, int max_nz,	const int *JA, const double *AS, const double *val_vec,	double *val_res);
