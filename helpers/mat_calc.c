#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
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
    
    cblas_domatcopy(
        CblasRowMajor, CblasTrans, 
        original->rows, original->cols, 
        1.0,
        original->data,
        original->cols, 
        resMatrix->data,
        resMatrix->cols
    );
}

// Sums 2 matrices and the result will be stored in resMatrix
void sumMatrices(Matrix* mat1, Matrix* mat2, Matrix* resMatrix) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        printf("ERROR: Different matrices sizes");
        exit(EXIT_FAILURE);
    }
    
    u32 size = mat1->rows * mat1->cols;
    cblas_dcopy(size, mat2->data, 1, resMatrix->data, 1);
    cblas_daxpy(size, 1.0, mat1->data, 1, resMatrix->data, 1);
}

// Multiply 2 matrices and the result will be stored in resMatrix
void matrixProduct(Matrix* mat1, Matrix* mat2, Matrix* resMatrix) {
    if (mat1->cols != mat2->rows) {
        printf("ERROR: Cols are not equal to rows");
        exit(EXIT_FAILURE);
    }
    
    resMatrix->rows = mat1->rows;
    resMatrix->cols = mat2->cols;
    if (resMatrix->data == NULL) resMatrix->data = (f64*)malloc(resMatrix->rows * resMatrix->cols * sizeof(f64));
    
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        mat1->rows, mat2->cols, mat1->cols, 
        1.0,
        mat1->data,
        mat1->cols, 
        mat2->data,
        mat2->cols, 
        0.0,
        resMatrix->data,
        resMatrix->cols
    );
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
    
    if (resMatrix->data != NULL) {
        memcpy(resMatrix->data, bias->data, resMatrix->rows * resMatrix->cols * sizeof(f64));
    } else {
        resMatrix->rows = bias->rows;
        resMatrix->cols = bias->cols;
        resMatrix->data = (f64*)malloc(resMatrix->rows * resMatrix->cols * sizeof(f64));
        memcpy(resMatrix->data, bias->data, resMatrix->rows * resMatrix->cols * sizeof(f64));
    }
    
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans,
        mat1->rows, mat1->cols,
        1.0,
        mat1->data,
        mat2->rows * mat2->cols,
        mat2->data,
        1,
        1.0,
        resMatrix->data,
        1
    );
}