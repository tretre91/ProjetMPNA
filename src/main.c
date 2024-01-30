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
#include <cblas.h>
#include <lapacke.h>

#include "argtable3.h"
#include "eig.h"


enum failed_operation {
    LU,
    INV,
    EIG
} FAILED_OP;

void load_mtx(double * m, int MATRIX_SIZE, FILE* file) {
    int row, col;
    double value;

    // Initialize the matrix with zeros
    for (row = 0; row < MATRIX_SIZE; ++row) {
        for (col = 0; col < MATRIX_SIZE; ++col) {
             m[row * MATRIX_SIZE + col]= 0.0;
        }
    }

    // Read data from the file and fill the matrix
    while (fscanf(file, "%d %d %lf", &row, &col, &value) == 3) {
        if (row <= MATRIX_SIZE && col <= MATRIX_SIZE) {
            m[( row-1)*MATRIX_SIZE + col- 1] = value;
        } else {
            fprintf(stderr, "Invalid row or column index in the file.\n");
        }
    }
}


int load_test_matrix_A(size_t n, double * A ){
    for ( size_t i = 0; i < n; i ++) {
        for ( size_t j = 0; j < n; j ++) {
            A[ i * n + j ] = 0;
        }
    }

    for ( size_t j = 0; j < n; j ++) {
        for ( size_t i = 0; i < j; i ++) {
            A[ i * n + j] = n + 1 - j;
        }
    }
    
    for ( size_t j = 0; j < n-1; j ++) {
        for ( size_t i = j+1; i < n; i ++) {
            A[ i * n + j] = n + 1 - i;
        }
    }
    return 0;
}

int load_test_matrix_B(size_t n, double * A ){

    for ( size_t i = 0; i < n; i ++) {
        for ( size_t j = 0; j < n; j ++) {
            A[ i * n + j ] = 0;
        }
    }
    double a = 3;
    double b = 6;

    for ( size_t i = 1; i < n-1; i ++) {
        for ( size_t j = 1; j < n-1; j ++) {
            A[ i * n + i - 1] = b;
            A[ i * n + i    ] = a;
            A[ i * n + i + 1] = b;
        }
    }
    
    A[0] = a; A[1] = b;
    A[n*n -2] = b; A[n*n-1] = a;
    
    return 0;
}


double* read_matrix(const char* filename, int* n) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    fscanf(file, " %d ", n);

    double* matrix;
    posix_memalign ((void**)&matrix, 32, *n * *n * sizeof(*matrix));

    int len = strlen(filename);
    if (len > 4 && strcmp(&filename[len - 4], ".mtx") == 0) {
        load_mtx(matrix, *n, file);
    } else {
        for (int i = 0; i < *n; i++) {
            for (int j = 0; j < *n; j++) {
                fscanf(file, " %lf ", &matrix[i * *n + j]);
            }
        }
    }

    fclose(file);

    return matrix;
}

// TODO: remove
void print_matrix_file(FILE* file, int nb_rows, int nb_columns, double* matrix) {
    for (int i = 0; i < nb_rows; i++) {
        for (int j = 0; j < nb_columns; j++) {
            // printf("%g ", matrix[j * nb_rows + i]);
            fprintf(file, "%.23e ", matrix[j * nb_rows + i]);
        }
        fprintf(file, "\n");
    }
}

void print_matrix(FILE* file, int nb_rows, int nb_columns, double* matrix) {
    for (int i = 0; i < nb_rows; i++) {
        for (int j = 0; j < nb_columns; j++) {
            // printf("%g ", matrix[j * nb_rows + i]);
            fprintf(file, "%.23e ", matrix[j * nb_rows + i]);
        }
        fprintf(file, "\n");
    }
}

void print_eigvals(FILE* file, int N, const double* eigvals_re, const double* eigvals_im) {
    for (int i = 0; i < N; i++) {
        fprintf(file, "%.5e %+.5ei\n", eigvals_re[i], eigvals_im[i]);
    }
}

#ifdef USE_CBLAS
double dot_product(int N, const double* restrict x, const double* restrict y) {

    return cblas_ddot(N, x, 1, y, 1);
}
#else
double dot_product(int N, const double* restrict x, const double* restrict y) {
    double res = 0;
    for (int i = 0; i < N; i++) {
        res += x[i] * y[i];
    }
    return res;
}
#endif



void matvec_product(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* y) {
    for (int i = 0; i < M; i++) {
        y[i] = dot_product(N, &A[i * lda], x);
    }
}

void omp_matvec_product(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* restrict y) {
#pragma omp for
    for (int i = 0; i < M; i++) {
        y[i] = dot_product(N, &A[i * lda], x);
    }
}



#ifdef USE_CBLAS
void matvec_product_col_major(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* restrict y) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1, A, M, x, 1, 0, y, 1);
    // cblas_dgemv(d, d, M, N, 1, A, N, X, 1, 0, y, 1);
}
#else

void matvec_product_col_major(int M, int N, const double* restrict A, const int lda, const double* restrict x, double* restrict y) {
    memset(y, 0, M * sizeof(*y));
    for (int i = 0; i < N; i++) {
        const double* column = &A[i * lda];
        const double val = x[i];
        for (int j = 0; j < M; j++) {
            y[j] += val * column[j];
        }
    }
}
#endif

double residual ;
double compute_residual(int N, const double*  restrict A, int lda, double eigval, const double* restrict eigvec) {
#pragma omp single 
    residual = 0.f;

#pragma omp for reduction(+:residual)
    for (int i = 0; i < N; i++) {
        double x = dot_product(N, &A[i * lda], eigvec) - eigval * eigvec[i];
        residual += x * x;
    }

    return sqrt(residual);
}


double compute_complex_residual(int N, const double* restrict A, int lda, double eigval_re, double eigval_im, const double* restrict eigvec_re, const double* restrict eigvec_im) {
#pragma omp single 
    residual = 0.f;

#pragma omp for reduction(+:residual)
    for (int i = 0; i < N; i++) {
        double x_re = dot_product(N, &A[i * lda], eigvec_re) - (eigval_re * eigvec_re[i] - eigval_im * eigvec_im[i]);
        double x_im = dot_product(N, &A[i * lda], eigvec_im) - (eigval_re * eigvec_im[i] + eigval_im * eigvec_re[i]);

        residual += x_re * x_re + x_im * x_im;
    }

    return sqrt(residual);
}

typedef struct prr_ret_type {
    double residual;
    double* eigvals_re;
    double* eigvals_im;
    double* eigvecs;
} prr_ret_type;

int error=0;

prr_ret_type prr(int N, const double* restrict A, int lda, const double* restrict y0, int s, int m, double epsilon, int max_iterations, int verbose) {
    // double Vm[N * (m + 1)];
    // double B_m1[m * m];
    // double B_m[m * m];
    // double C[2 * m];
    double * Vm;
    double * B_m1;
    double * B_m;
    double * C;

    posix_memalign ((void**)&Vm, 32, N * (m + 1) * sizeof(*Vm) );
    posix_memalign ((void**)&B_m1, 32, m * m * sizeof(*B_m1) );
    posix_memalign ((void**)&B_m, 32, m * m * sizeof(*B_m) );
    posix_memalign ((void**)&C, 32, 2 * m * sizeof(*C) );

    double* eigvals_re;
    posix_memalign ((void**)&eigvals_re, 32, m * sizeof(*eigvals_re) );

    double* eigvals_im;
    posix_memalign ((void**)&eigvals_im, 32, m * sizeof(*eigvals_im) );

    double * eigvecs;
    posix_memalign ((void**)&eigvecs, 32, m * m * sizeof(*eigvecs) );

    double* q;
    posix_memalign ((void**)&q, 32, (m + 1) * N * sizeof(*q) );

    double max_residual = DBL_MIN;

    double* conjugate_eigvec;
    double cur_residual;
    bool is_complex;


    memcpy(Vm, y0, N * sizeof(double));
    const double norm_y0 = 1 / sqrt(dot_product(N, Vm, Vm));
#pragma omp parallel
    {

#pragma omp for
        for (int i = 0; i < N; i++) {
            Vm[i] *= norm_y0;
        }

        for (int it = 0; it < max_iterations; it++) {
            // étape 1 : constituer les matrices B_m-1, B_m et Vm

            for (int i = 1; i < m + 1; i++) {
                omp_matvec_product(N, N, A, lda, &Vm[(i - 1) * N], &Vm[i * N]);
                // const double norm = 1 / sqrt(dot_product(N, &Vmtmp[i * N], &Vmtmp[i * N]));
                // for (int j = 0; j < N; j++) {
                //     Vmtmp[i * N + j] *= norm;
                // }
            }

#pragma omp for
            for (int i = 0; i < m; i++) {
                C[2 * i] = dot_product(N, &Vm[i * N], &Vm[i * N]);
                C[2 * i + 1] = dot_product(N, &Vm[i * N], &Vm[(i + 1) * N]);
            }

#pragma omp for
            for (int i = 0; i < m; i++) {
                B_m1[i * m + i] = C[i + i];
                B_m[i * m + i] = C[i + i + 1];
                for (int j = i + 1; j < m; j++) {
                    B_m1[i * m + j] = C[i + j];
                    B_m1[j * m + i] = B_m1[i * m + j];

                    B_m[i * m + j] = C[i + j + 1];
                    B_m[j * m + i] = B_m[i * m + j];
                }
            }

#pragma omp single
            {
                // étape 2 : résolution dans le sous-espace

                int32_t* ipiv;
                posix_memalign ((void**)&ipiv, 32, m * sizeof(*ipiv));

                int info = LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'U', m, B_m1, m, ipiv);
                if (info != 0 && error == 0) {
                    FAILED_OP = LU;
                    error = 1;
                    fprintf(stderr, "LU factorization failed with info = %d\n", info);
                    // TODO: return error
                }
                info = LAPACKE_dsytri(LAPACK_COL_MAJOR, 'U', m, B_m1, m, ipiv);
                if (info != 0 && error == 0) {
                    FAILED_OP = INV;
                    error = 1;
                    fprintf(stderr, "Matrix inverse failed with info = %d\n", info);
                    // TODO: return error
                }
                free(ipiv);

                double* F_m;
                posix_memalign ((void**)&F_m, 32, m * m * sizeof(*F_m));

                cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, m, m, 1, B_m1, m, B_m, m, 0, F_m, m);
                // if (verbose ) {
                //     printf("\nFm =\n");
                //     print_matrix(m, m, F_m);
                // }
                
                // could be par
                info = sorted_eigvals(m, F_m, m, eigvals_re, eigvals_im, eigvecs);
                free(F_m);

                if (info != 0 && error==0) {
                    error =1;
                    FAILED_OP = EIG;
                    fprintf(stderr, "Eigenvalue compuatation failed with info = %d\n", info);
                    // TODO: return error
                }
            }


                // étape 3 : retour dans l'espace de départ
#pragma omp for
            for (int i = 0; i < m; i++) {
                matvec_product_col_major(N, m, Vm, N, eigvecs + i * m, q + i * N);
            }



// étape 4 : calcul de l'erreur
#pragma omp single
            {
                max_residual = DBL_MIN;
                is_complex = false;
                posix_memalign ((void**)&conjugate_eigvec, 32, N * sizeof(conjugate_eigvec));
            }
// #pragma omp single
//             {

                for (int i = 0; i < s; i++) {
                    if (is_complex) {
#pragma omp barrier // je sais pas pourquoi il faut une omp barrier mais bon ca fonctionne avec

                        double local_residual = compute_complex_residual(N, A, lda, eigvals_re[i], eigvals_im[i], &q[(i - 1) * N], conjugate_eigvec);
#pragma omp single
                        {
                            cur_residual = local_residual;
                            is_complex = false;
                        }

                    } else if (eigvals_im[i] != 0.0) {

                        const double* eigvec_im = &q[(i + 1) * N];
#pragma omp single
                        {
                            for (int j = 0; j < N; j++) {
                                conjugate_eigvec[j] = -eigvec_im[j];
                            }
                        }
                        double local_residual = compute_complex_residual(N, A, lda, eigvals_re[i], eigvals_im[i], &q[i * N], eigvec_im);
#pragma omp single
                        {
                            cur_residual = local_residual;
                            is_complex = true;
                        }

                    } else {
                         double local_residual = compute_residual(N, A, lda, eigvals_re[i], &q[i * N]);
#pragma omp single
                        {
                            cur_residual = local_residual;
                        }
                    }

                }
#pragma omp single
            {
                    if (cur_residual > max_residual) {
                        max_residual = cur_residual;
                    }

                free(conjugate_eigvec);

                printf("%d %g\n", it, max_residual);

                // cancel break for now
                // if (max_residual < epsilon && false) {
                //     break;
                // }

                // redémarage
                // note : directly update &Vm[0 * N]

                memset(Vm, 0, N * sizeof(*Vm));
            }
#pragma omp single
            {

                bool prev_is_complex = false;
                for (int i = 0; i < s; i++) {
                    double coef = rand() / (double)RAND_MAX;
                    if (prev_is_complex) {
                        double* eigvec_re = &q[i * N];
                        double* eigvec_im = &q[(i + 1) * N];
                        for (int j = 0; j < N; j++) {
                            Vm[j] += coef * cabs(eigvec_re[j] + I * eigvec_im[j]);
                        }
                        prev_is_complex = false;
                    } else if (eigvals_im[i] != 0) {
                        double* eigvec_re = &q[i * N];
                        double* eigvec_im = &q[(i + 1) * N];
                        for (int j = 0; j < N; j++) {
                            Vm[j] += coef * cabs(eigvec_re[j] + I * eigvec_im[j]);
                        }
                        prev_is_complex = true;
                    } else {
                        double* eigvec = &q[i * N];
                        for (int j = 0; j < N; j++) {
                            Vm[j] += coef * fabs(eigvec[j]);
                        }
                    }
                } //end restart


            } // end single
// #pragma omp barrier
        if(error==1)break;
        } // end for
    } // end pragma parallel

    prr_ret_type ret = {max_residual, eigvals_re, eigvals_im, q};
    free(eigvecs);
    free( Vm );
    free( B_m1 );
    free( B_m );
    free( C );


    return ret;
}


int main(int argc, char* argv[]) {
    struct arg_int* s = arg_int1("s", "nb-eigvals", "INT", "Number of eigenvalues-eigenvector pairs to approximate");
    struct arg_int* m = arg_int1("m", "subspace-size", "INT", "Dimensions of the projected subspace, should be greater than s");
    struct arg_dbl* epsilon =
      arg_dbl0("e", "epsilon", "FLOAT", "Tolerance parameter for the maximum error of projected eigenvalues/vectors pairs. Defaults to 1e-6");
    struct arg_int* nb_iterations = arg_int0("i", "iterations", "INT", "Maximum number of iterations. Defaults to 1000");
    struct arg_lit* verbose = arg_lit0("v", "verbose", "If present, additional information will be printed to stdout");
    struct arg_lit* help = arg_lit0("h", "help", "Display this help message");
    struct arg_file* matrix_file = arg_file1(NULL, NULL, "MATRIX_FILE", "A file containing the matrix");
    struct arg_end* end = arg_end(5);

    void* argtable[] = {s, m, epsilon, nb_iterations, verbose, help, matrix_file, end};

    const int nb_opts = sizeof argtable / sizeof argtable[0];

    if (arg_nullcheck(argtable) != 0) {
        fprintf(stderr, "Argument parsing failed\n");
        arg_freetable(argtable, nb_opts);
        return 1;
    }

    epsilon->dval[0] = 1e-6;
    nb_iterations->ival[0] = 1000;

    int nb_errors = arg_parse(argc, argv, argtable);

    if (help->count > 0) {
        printf("Usage: %s ", argv[0]);
        arg_print_syntax(stdout, argtable, "\n\n");
        arg_print_glossary(stdout, argtable, "  %-30s %s\n");
        arg_freetable(argtable, nb_opts);
        return 0;
    }

    if (nb_errors > 0) {
        arg_print_errors(stderr, end, argv[0]);
        arg_freetable(argtable, nb_opts);
        return 1;
    }
   
    // orig : -------------------
    int n;
    double* matrix = read_matrix(matrix_file->filename[0], &n);


    /// for  benchmark : -------------------
    // m->ival[0] = 100;
    // int n = 3000;
    // double* matrix = malloc(n * n * sizeof(*matrix));
    // load_test_matrix_B(n, matrix);

    //-------------------------------
    // int n = 1138;
    // double* matrix = malloc(n * n * sizeof(*matrix));
    // load_mtx( matrix, n, "data/1138_bus.mtx" );
    
    // int n = 9540;
    // double* matrix = malloc(n * n * sizeof(*matrix));
    // load_mtx( matrix, n, "../data/coater2_9540.mtx" );

    // int n = 362;
    // double* matrix = malloc(n * n * sizeof(*matrix));
    // load_mtx( matrix, n, "../data/plat362.mtx.gz" );

    srand(0);
    double* y0;
    posix_memalign ((void**)&y0, 32, n  * sizeof(*y0));

    for (int i = 0; i < n; i++) {
        y0[i] = rand() / (double)RAND_MAX;
    }

    prr_ret_type result = prr(n, matrix, n, y0, s->ival[0], m->ival[0], epsilon->dval[0], nb_iterations->ival[0], verbose->count > 0);

    print_eigvals(stderr, s->ival[0], result.eigvals_re, result.eigvals_im);
    // if (verbose->count > 0) {
    //     print_matrix(n, s->ival[0], result.eigvecs);
    // }

    free(matrix);
    free(y0);
    free(result.eigvals_re);
    free(result.eigvals_im);
    free(result.eigvecs);
    arg_freetable(argtable, nb_opts);
}
