#include "mat_calc.h"
#include "types.h"

#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))

// Multiply 2 matrices
void matrixProduct(Matrix* matrix1, Matrix* matrix2, Matrix* endMatrix) {
    if (matrix1->cols != matrix2->rows) return;
    
}