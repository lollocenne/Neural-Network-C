#include "cost_functions.h"
#include "../types.h"

f64 squaredError(f64 expected, f64 predicted) {return (expected - predicted) * (expected - predicted) / 2;}

f64 squaredErrorDerivate(f64 expected, f64 predicted) {return predicted - expected;}