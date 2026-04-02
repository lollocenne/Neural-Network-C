#include <stdlib.h>
#include <math.h>
#include "cost_functions.h"
#include "../types.h"

f64 absoluteError(f64 expected, f64 predicted) {return abs(expected - predicted);}
f64 squaredError(f64 expected, f64 predicted) {return (expected - predicted) * (expected - predicted) / 2;}
f64 logCosh(f64 expected, f64 predicted) {return log(cosh(expected - predicted));}

f64 squaredErrorDerivate(f64 expected, f64 predicted) {return predicted - expected;}
f64 absoluteErrorDerivate(f64 expected, f64 predicted) {return predicted > expected ? 1 : -1;}
f64 logCoshDerivate(f64 expected, f64 predicted) {return tanh(predicted - expected);}