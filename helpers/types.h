#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef int32_t  i32;

typedef uint32_t u32;

typedef float    f32;
typedef double   f64;


typedef f64 (*funcOneParam)(f64);
typedef f64 (*funcTwoParam)(f64, f64);

typedef struct {
    u32 rows;
    u32 cols;
    f64* data;
} Matrix;

typedef struct {
    Matrix* neurons;
    Matrix* zs; // the value before applying the activation function
    Matrix* weights;
    Matrix* bias;
    Matrix* momentumW;
    Matrix* momentumB;
    Matrix* signalError; // the error signal for each neuron
    funcOneParam actFunction;
    funcOneParam derActFunction;
} Layer;

#endif