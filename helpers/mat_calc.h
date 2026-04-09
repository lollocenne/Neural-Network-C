#ifndef MAT_CALC_H
#define MAT_CALC_H

#include "types.h"

typedef struct {
    u32 rows;
    u32 cols;
    f64* data;
} Matrix;

#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))

void transposeMatrix(Matrix* original, Matrix* resMatrix);
void sumMatrices(Matrix* mat1, Matrix* mat2, Matrix* resMatrix);
void matrixProduct(Matrix* mat1, Matrix* mat2, Matrix* resMatrix);
void multiplyMatrix(Matrix* mat, f64 num, Matrix* resMatrix);
void matrixProductWithBias(Matrix* mat1, Matrix* mat2, Matrix* bias, Matrix* resMatrix);

#endif