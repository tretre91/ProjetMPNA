#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



void load_mtx(double * m, int MATRIX_SIZE, FILE* file) ;
int load_test_matrix_A(size_t n, double * A );
int load_test_matrix_B(size_t n, double * A );
double* read_matrix(const char* filename, int* n) ;
void print_matrix_file(FILE* file, int nb_rows, int nb_columns, double* matrix);
void print_matrix(FILE* file, int nb_rows, int nb_columns, double* matrix);

