#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// #define LAPACK_DISABLE_NAN_CHECK
#if __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif


#include "argtable3.h"
#include "eig.h"
#include "mat.h"

void print_eigvals(FILE* file, int N, const double* eigvals_re, const double* eigvals_im);
double dot_product(int N, const double* restrict x, const double* restrict y) ;
void matvec_product(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* y) ;
void omp_matvec_product(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* restrict y) ;
void matvec_product_col_major(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* restrict y) ;
double compute_residual(int N, const double*  restrict A, int lda, double eigval, const double* restrict eigvec);
double compute_complex_residual(int N, const double* restrict A, int lda, double eigval_re, double eigval_im, const double* restrict eigvec_re, const double* restrict eigvec_im) ;

typedef struct prr_ret_type {
    double residual;
    double* eigvals_re;
    double* eigvals_im;
    double* eigvecs;
} prr_ret_type;
prr_ret_type prr(int N, const double* restrict A, int lda, const double* restrict y0, int s, int m, double epsilon, int max_iterations, int verbose) ;
