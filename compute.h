double sequential_csr(struct csrFormat *csr, struct vector *vec, struct vector *res);
double sequential_ellpack(struct ellpackFormat *ellpack, struct vector *vec, struct vector *res);

double csr_simd_reduction(struct csrFormat *csr, struct vector *vec, struct vector *res, int nz);

double ellpack_simd_reduction(struct ellpackFormat *ellpack, struct vector *vec, struct vector *res, int nz);


