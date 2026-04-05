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

#endif