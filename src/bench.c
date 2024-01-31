#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <omp.h>

#include "argtable3.h"
#include "eig.h"
#include "mat.h"
#include "prr.h"

void print_double(double *restrict a, int n) {
    for (int i = 0; i < n * n; i++)
        printf("%lf%c", a[i], ((i + 1) % n) ? ' ' : '\n');

    printf("\n");
}

//


//
void sort_double(double *restrict a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (a[i] > a[j]) {
                double tmp = a[i];

                a[i] = a[j];
                a[j] = tmp;
            }
}

//
double mean_double(double *restrict a, int n) {
    double m = 0.0;

    for (int i = 0; i < n; i++)
        m += a[i];

    m /= (double)n;

    return m;
}

//
double stddev_double(double *restrict a, int n) {
    double d = 0.0;
    double m = mean_double(a, n);

    for (int i = 0; i < n; i++)
        d += (a[i] - m) * (a[i] - m);

    d /= (double)(n - 1);

    return sqrt(d);
}

double gettime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e0 + ts.tv_nsec*1e-9;
}

int run_benchmark(int n, const double* restrict matrix, int lda, const double* restrict y0, int s, int m, double epsilon, int max_iterations, int verbose) {

    //
    int r = 3;
    int MAX_SAMPLES = 3;
    double elapsed = 0.0;
    double t1, t2;
    double samples[MAX_SAMPLES];

    //
    for (int i = 0; i < MAX_SAMPLES; i++) {
        do {
            // clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
            t1 = gettime();

            for (int j = 0; j < r; j++){
                prr_ret_type result = prr(n, matrix, lda, y0, s, m, epsilon, max_iterations, verbose);
            }

            t2 = gettime();
            // clock_gettime(CLOCK_MONOTONIC_RAW, &t2);

            elapsed = (double)(t2 - t1) / (double)r;
        } while (elapsed <= 0.0);

        samples[i] = elapsed;
    }
    
    //
    sort_double(samples, MAX_SAMPLES);

    //
    double min = samples[0];
    double max = samples[MAX_SAMPLES - 1];
    double mean = mean_double(samples, MAX_SAMPLES);
    double dev = stddev_double(samples, MAX_SAMPLES);

    // Size in MiB / time in seconds

    //
    // printf("%10s; %15.3lf; %15.3lf; %15.3lf; %10llu; %10llu; %15.3lf; %15.3lf; "
    //        "%15.3lf; %15.3lf (%6.3lf %%); %10.3lf; %10s\n",
    //        "prr",
    //        3 * (double)0, // 3 matices
    //        3 * (double)0, // 3 matrices
    //        3 * (double)0, // 3 matrices
    //        n, r, min, max, mean, dev, (dev * 100.0 / mean), (double)0, "prr");

    // printf("nthreads;n;m;s;mean;dev;version");
    int nthreads = omp_get_max_threads();    
    printf("%d;%d;%d;%d;%f;%f;%s\n", nthreads,n, m, s, mean, dev, 
#ifdef USE_CBLAS
           "blas"
#else
           "no_blas"
#endif
           );


    
    return 0;
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



    srand(0);
    double* y0;
    posix_memalign ((void**)&y0, 32, n  * sizeof(*y0));

    for (int i = 0; i < n; i++) {
        y0[i] = rand() / (double)RAND_MAX;
    }


    // printf("%10s; %15s; %15s; %15s; %10s; %10s; %15s; %15s; %15s; %26s; %10s; %10s\n",
    //        "titre", "KiB", "MiB", "GiB", "n", "r", "min", "max", "mean",
    //        "stddev (%)", "MiB/s", "titre");

    run_benchmark(n, matrix, n, y0, s->ival[0], m->ival[0], epsilon->dval[0], nb_iterations->ival[0], verbose->count > 0);



    free(matrix);
    free(y0);
    arg_freetable(argtable, nb_opts);
}



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

 
