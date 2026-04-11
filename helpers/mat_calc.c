#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mat_calc.h"
#include "types.h"

// Use the entire matrix struct
#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))

// Use just the data of the matrix
#define GET_ARRAY_ELEMENT(matrixData, matrixCols, row, col) (matrixData[(row) * matrixCols + (col)])
#define SET_ARRAY_ELEMENT(matrixData, matrixCols, row, col, val) (matrixData[(row) * matrixCols + (col)] = (val))


// Get the transpose of a matrix and the result will be stored in resMatrix
void transposeMatrix(Matrix* original, Matrix* resMatrix) {
    if (resMatrix->data == NULL) resMatrix->data = (f64*)malloc(original->rows * original->cols * sizeof(f64));
    resMatrix->rows = original->cols;
    resMatrix->cols = original->rows;
    
    f64* originalData = original->data;
    f64* resData = resMatrix->data;
    u32 originalRows = original->rows;
    u32 originalCols = original->cols;
    for (u32 i = 0; i < originalRows; i++) {
        for (u32 j = 0; j < originalCols; j++) {
            SET_ARRAY_ELEMENT(resData, originalRows, j, i, GET_ARRAY_ELEMENT(originalData, originalCols, i, j));
        }
    }
}

// Sums 2 matrices and the result will be stored in resMatrix
void sumMatrices(Matrix* mat1, Matrix* mat2, Matrix* resMatrix) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        printf("ERROR: Different matrices sizes");
        exit(EXIT_FAILURE);
    }
    
    u32 size = mat1->rows * mat1->cols;
    f64* a = mat1->data;
    f64* b = mat2->data;
    f64* r = resMatrix->data;
    #pragma omp parallel for schedule(static)
    for (u32 i = 0; i < size; i++) {
        r[i] = a[i] + b[i];
    }
}

// Multiply 2 matrices and the result will be stored in resMatrix
void matrixProduct(Matrix* mat1, Matrix* mat2, Matrix* resMatrix) {
    if (mat1->cols != mat2->rows) {
        printf("ERROR: Cols are not equal to rows");
        exit(EXIT_FAILURE);
    }
    
    resMatrix->rows = mat1->rows;
    resMatrix->cols = mat2->cols;
    if (resMatrix->data != NULL) {
        memset(resMatrix->data, 0, resMatrix->rows * resMatrix->cols * sizeof(f64));
    } else {
        resMatrix->data = (f64*)calloc(resMatrix->rows * resMatrix->cols, sizeof(f64));
    }
    
    f64* mat1Data = mat1->data;
    f64* mat2Data = mat2->data;
    f64* resData = resMatrix->data;
    u32 mat1Rows = mat1->rows;
    u32 mat1Cols = mat1->cols;
    u32 mat2Cols = mat2->cols;
    f64 tempMat;
    #pragma omp parallel for schedule(static)
    for (u32 i = 0; i < mat1Rows; i++) {
        for (u32 k = 0; k < mat1Cols; k++) {
            tempMat = GET_ARRAY_ELEMENT(mat1Data, mat1Cols, i, k);
            for (u32 j = 0; j < mat2Cols; j++) {
                GET_ARRAY_ELEMENT(resData, mat2Cols, i, j) += tempMat * GET_ARRAY_ELEMENT(mat2Data, mat2Cols, k, j);
            }
        }
    }
}

// Multiply a matrix with a number and the result will be stored in resMatrix
void multiplyMatrix(Matrix* mat, f64 num, Matrix* resMatrix) {
    f64* matData = mat->data;
    f64* resData = resMatrix->data;
    for (u32 i = 0; i < mat->rows * mat->cols; i++) {
        resData[i] = matData[i] * num;
    }
}

// Multiply 2 matrices and sum bias, result stored in resMatrix
// Equivalent to: matrixProduct(mat1, mat2, res) + sumMatrices(res, bias, res) but in a single loop, avoiding a second pass over the data
void matrixProductWithBias(Matrix* mat1, Matrix* mat2, Matrix* bias, Matrix* resMatrix) {
    if (mat1->cols != mat2->rows) {
        printf("ERROR: Cols are not equal to rows");
        exit(EXIT_FAILURE);
    }
    
    resMatrix->rows = mat1->rows;
    resMatrix->cols = mat2->cols;
    
    u32 totalSize = resMatrix->rows * resMatrix->cols;
    
    if (resMatrix->data != NULL) {
        memcpy(resMatrix->data, bias->data, totalSize * sizeof(f64));
    } else {
        resMatrix->data = (f64*)malloc(totalSize * sizeof(f64));
        memcpy(resMatrix->data, bias->data, totalSize * sizeof(f64));
    }
    
    f64* mat1Data = mat1->data;
    f64* mat2Data = mat2->data;
    f64* resData  = resMatrix->data;
    u32 mat1Rows = mat1->rows;
    u32 mat1Cols = mat1->cols;
    u32 mat2Cols = mat2->cols;
    #pragma omp parallel for schedule(static)
    for (u32 i = 0; i < mat1Rows; i++) {
        f64 tempMat;
        for (u32 k = 0; k < mat1Cols; k++) {
            tempMat = GET_ARRAY_ELEMENT(mat1Data, mat1Cols, i, k);
            for (u32 j = 0; j < mat2Cols; j++) {
                GET_ARRAY_ELEMENT(resData, mat2Cols, i, j) += tempMat * GET_ARRAY_ELEMENT(mat2Data, mat2Cols, k, j);
            }
        }
    }
}