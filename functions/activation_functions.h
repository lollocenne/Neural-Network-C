#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "../types.h"

f64 identity(f64 x);
f64 binaryStep(f64 x);
f64 sigmoid(f64 x);
f64 tanh(f64 x);
f64 relu(f64 x);

f64 derivativeIdentity(f64 x);
f64 derivativeBinaryStep(f64 x);
f64 derivativeSigmoid(f64 x);
f64 derivativeTanh(f64 x);
f64 derivativeRelu(f64 x);

#endif