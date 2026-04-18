#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "../helpers/types.h"

f64 identity(f64 x);
f64 binaryStep(f64 x);
f64 sigmoid(f64 x);
f64 tanh(f64 x);
f64 relu(f64 x);
f64 leakyRelu(f64 x);
f64 softPlus(f64 x);
f64 gaussian(f64 x);
f64 sinusoid(f64 x);
void softmax(Layer* l, u32 size);

f64 derivativeIdentity(f64 x);
f64 derivativeBinaryStep(f64 x);
f64 derivativeSigmoid(f64 x);
f64 derivativeTanh(f64 x);
f64 derivativeRelu(f64 x);
f64 derivativeLeakyRelu(f64 x);
f64 derivativeSoftPlus(f64 x);
f64 derivativeGaussian(f64 x);
f64 derivativeSinusoid(f64 x);

#endif