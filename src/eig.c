#if __has_include(<openblas/cblas.h>)
#include <openblas/lapacke.h>
#else
#include <lapacke.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

double complex_norm2(double re, double im) {
    return re * re + im * im;
}

double complex_norm(double re, double im) {
    return sqrt(re * re + im * im);
}

static double* _sort_eigvals_re = NULL;
static double* _sort_eigvals_im = NULL;

// int compare(const void* a, const void* b) {
//     const int lhs = *(int*)a;
//     const int rhs = *(int*)b;
//
//     const double norm_lhs = complex_norm(_sort_eigvals_re[lhs], _sort_eigvals_im[lhs]);
//     const double norm_rhs = complex_norm(_sort_eigvals_re[rhs], _sort_eigvals_im[rhs]);
//
//     const int cmp = 100 * (norm_rhs - norm_lhs);
//     // return cmp;
//     return cmp != 0 ? cmp : lhs - rhs;
// }

int compare(const void* a, const void* b) {
    const int lhs = *(int*)a;
    const int rhs = *(int*)b;

    const double norm_lhs = complex_norm2(_sort_eigvals_re[lhs], _sort_eigvals_im[lhs]);
    const double norm_rhs = complex_norm2(_sort_eigvals_re[rhs], _sort_eigvals_im[rhs]);
    
    const double diff = norm_rhs - norm_lhs;
    return fabs(diff) < 1e-15 ? lhs - rhs : (int)copysign(2, diff);
    // if (norm_lhs > norm_rhs) {
    //     return -1;
    // } else if (norm_lhs < norm_rhs) {
    //     return 1;
    // } else {
    //     return lhs - rhs;
    // }

    // const long cmp = lround(norm_rhs - norm_lhs);
    //
    // return cmp != 0 ? cmp : lhs - rhs; // TODO: restrictive cast from long to int
}

int sorted_eigvals(int N, double* A, int lda, double* eigvals_re, double* eigvals_im, double* eigvecs) {
    double eigvals_re_work[N];
    double eigvals_im_work[N];
    double eigvecs_work[N * N];

    int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A, lda, eigvals_re_work, eigvals_im_work, eigvecs_work, N, eigvecs_work, N);
    if (info != 0) {
        return info;
    }

    _sort_eigvals_re = eigvals_re_work;
    _sort_eigvals_im = eigvals_im_work;

    int* indices = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }

    qsort(indices, N, sizeof(int), compare);

    for (int i = 0; i < N; i++) {
        const int idx = indices[i];
        eigvals_re[i] = eigvals_re_work[idx];
        eigvals_im[i] = eigvals_im_work[idx];
        memcpy(eigvecs + i * N, eigvecs_work + idx * N, N * sizeof(*eigvecs));
    }

    free(indices);

    return 0;
}
