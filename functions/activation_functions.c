#include <math.h>
#include "activation_functions.h"
#include "../types.h"

f64 identity(f64 x)             {return x;}
f64 binaryStep(f64 x)           {return x >= 0 ? 1 : 0;}
f64 sigmoid(f64 x)              {return 1 / (1 + exp(-x));}
f64 tanh(f64 x)                 {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));}
f64 relu(f64 x)                 {return x > 0 ? x : 0;}

f64 derivativeIdentity(f64 x)   {return 1;}
f64 derivativeBinaryStep(f64 x) {return 0;}
f64 derivativeSigmoid(f64 x)    {return sigmoid(x) * (1 - sigmoid(x));}
f64 derivativeTanh(f64 x)       {return 1 - tanh(x) * tanh(x);}
f64 derivativeRelu(f64 x)       {return x > 0 ? 1 : 0;}