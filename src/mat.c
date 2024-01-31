#include "mat.h"

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
