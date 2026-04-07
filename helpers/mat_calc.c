#include <stdlib.h>
#include <string.h>
#include "mat_calc.h"
#include "types.h"

#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))

// Get the transpose of a matrix and the result will be stored in resMatrix
void transposeMatrix(Matrix* original, Matrix* resMatrix) {
    if (resMatrix->data == NULL) resMatrix->data = (f64*)malloc(original->rows * original->cols * sizeof(f64));
    resMatrix->rows = original->cols;
    resMatrix->cols = original->rows;
    
    for (u32 i = 0; i < original->rows; i++) {
        for (u32 j = 0; j < original->cols; j++) {
            SET_MATRIX_ELEMENT(resMatrix, j, i, GET_MATRIX_ELEMENT(original, i, j));
        }
    }
}

// Sums 2 matrices and the result will be stored in resMatrix
void sumMatrices(Matrix* mat1, Matrix* mat2, Matrix* resMatrix) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) return;
    for (u32 i = 0; i < mat1->cols; i++) {
        for (u32 j = 0; j < mat1->rows; j++) {
            SET_MATRIX_ELEMENT(resMatrix, i, j, GET_MATRIX_ELEMENT(mat1, i, j) + GET_MATRIX_ELEMENT(mat2, i, j));
        }
    }
}

// Multiply 2 matrices and the result will be stored in resMatrix
void matrixProduct(Matrix* mat1, Matrix* mat2, Matrix* resMatrix) {
    if (mat1->cols != mat2->rows) return;
    resMatrix->rows = mat1->rows;
    resMatrix->cols = mat2->cols;
    if (resMatrix->data != NULL) {
        memset(resMatrix->data, 0, resMatrix->rows * resMatrix->cols * sizeof(f64));
    } else {
        resMatrix->data = (f64*)calloc(resMatrix->rows * resMatrix->cols, sizeof(f64));
    }
    
    for (u32 i = 0; i < mat1->rows; i++) {
        for (u32 j = 0; j < mat2->cols; j++) {
            for (u32 k = 0; k < mat1->cols; k++) {
                GET_MATRIX_ELEMENT(resMatrix, i, j) += GET_MATRIX_ELEMENT(mat1, i, k) * GET_MATRIX_ELEMENT(mat2, k, j);
            }
        }
    }
}