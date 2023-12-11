#include <lapacke.h>
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

    if (norm_lhs > norm_rhs) {
        return -1;
    } else if (norm_lhs < norm_rhs) {
        return 1;
    } else {
        return lhs - rhs;
    }
    // const long cmp = lround(norm_rhs - norm_lhs);
    //
    // return cmp != 0 ? cmp : lhs - rhs; // TODO: restrictive cast from long to int
}

int sorted_eigvals(int N, double* A, int lda, int limit, double** out_eigvals, double** out_eigvecs) {
    double* eigvals_re = malloc(N * sizeof(*eigvals_re));
    double* eigvals_im = malloc(N * sizeof(*eigvals_im));
    double* eigvecs = malloc(N * N * sizeof(*eigvecs));

    LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A, lda, eigvals_re, eigvals_im, eigvecs, N, eigvecs, N);

    _sort_eigvals_re = eigvals_re;
    _sort_eigvals_im = eigvals_im;

    int* indices = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }

    qsort(indices, N, sizeof(int), compare);

    // Si les valeurs propres limit-1 et limit forment une paire complexe conjuguée,
    // on inclut la valeur et le vecteur propre d'indice limit
    if (limit != N && eigvals_im[indices[limit - 1]] != 0.0) {
        bool cut = true;
        for (int i = limit - 2; i >= 0 && eigvals_im[indices[i]] != 0; i--) {
            cut = !cut;
        }
        if (cut) {
            limit += 1;
        }
    }

    double* sorted_eigvals = malloc(2 * limit * sizeof(double));
    double* sorted_eigvecs = malloc(N * limit * sizeof(double));

    for (int i = 0; i < limit; i++) {
        const int idx = indices[i];
        sorted_eigvals[i] = eigvals_re[idx];
        sorted_eigvals[limit + i] = eigvals_im[idx];
        memcpy(sorted_eigvecs + i * N, eigvecs + idx * N, N * sizeof(*eigvecs));
    }

    free(eigvals_re);
    free(eigvals_im);
    free(eigvecs);
    free(indices);

    *out_eigvals = sorted_eigvals;
    *out_eigvecs = sorted_eigvecs;

    return limit;
}
