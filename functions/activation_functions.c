#include <math.h>
#include "activation_functions.h"
#include "../helpers/types.h"

#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))


f64 identity(f64 x)             {return x;}
f64 binaryStep(f64 x)           {return x >= 0 ? 1 : 0;}
f64 sigmoid(f64 x)              {f64 expp = exp(-x); return 1 / (1 + expp);}
f64 tanh(f64 x)                 {f64 exp1 = exp(x), exp2 = 1/exp1; return (exp1 - exp2) / (exp1 + exp2);}
f64 relu(f64 x)                 {return x > 0 ? x : 0;}
f64 leakyRelu(f64 x)            {return x > 0 ? x : 0.01 * x;}
f64 softPlus(f64 x)             {return log(1 + exp(x));}
f64 gaussian(f64 x)             {return exp(-x * x);}
f64 sinusoid(f64 x)             {return sin(x);}
void softmax(Layer* l, u32 size) {
    f64 sum = 0;
    f64* zsData = l->zs->data;
    for (u32 i = 0; i < size; i++) {
        sum += exp(zsData[i]);
    }
    for (u32 i = 0; i < size; i++) {
        l->neurons->data[i] = exp(zsData[i]) / sum;
    }
}

f64 derivativeIdentity(f64 x)   {return 1;}
f64 derivativeBinaryStep(f64 x) {return 0;}
f64 derivativeSigmoid(f64 x)    {f64 s = sigmoid(x); return s * (1 - s);}
f64 derivativeTanh(f64 x)       {f64 t = tanh(x); return 1 - t * t;}
f64 derivativeRelu(f64 x)       {return x > 0 ? 1 : 0;}
f64 derivativeLeakyRelu(f64 x)  {return x > 0 ? 1 : 0.01;}
f64 derivativeSoftPlus(f64 x)   {return 1/(1 + exp(-x));}
f64 derivativeGaussian(f64 x)   {return -2*x*exp(-x * x);}
f64 derivativeSinusoid(f64 x)   {return cos(x);}