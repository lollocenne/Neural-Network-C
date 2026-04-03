#include <stdlib.h>
#include <math.h>
#include "cost_functions.h"
#include "../types.h"

#define EPSILON 0.000000000000001
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

f64 absoluteError(f64 expected, f64 predicted) {return fabs(expected - predicted);}
f64 squaredError(f64 expected, f64 predicted) {return (expected - predicted) * (expected - predicted) / 2;}
f64 logCosh(f64 expected, f64 predicted) {return log(cosh(expected - predicted));}
f64 crossEntropy(f64 expected, f64 predicted) {
    predicted = max(min(predicted, 1 - EPSILON), EPSILON);
    return -(expected * log(predicted) + (1 - expected) * log(1 - predicted));
}

f64 absoluteErrorDerivate(f64 expected, f64 predicted) {return predicted > expected ? 1 : -1;}
f64 squaredErrorDerivate(f64 expected, f64 predicted) {return predicted - expected;}
f64 logCoshDerivate(f64 expected, f64 predicted) {return tanh(predicted - expected);}
f64 crossEntropyDerivate(f64 expected, f64 predicted) {
    predicted = max(min(predicted, 1 - EPSILON), EPSILON);
    return -(expected / predicted) + ((1 - expected) / (1 - predicted));
}