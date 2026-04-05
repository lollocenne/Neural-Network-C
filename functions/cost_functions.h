#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "../helpers/types.h"

f64 absoluteError(f64 expected, f64 predicted);
f64 squaredError(f64 expected, f64 predicted);
f64 logCosh(f64 expected, f64 predicted);
f64 crossEntropy(f64 expected, f64 predicted);

f64 squaredErrorDerivate(f64 expected, f64 predicted);
f64 absoluteErrorDerivate(f64 expected, f64 predicted);
f64 logCoshDerivate(f64 expected, f64 predicted);
f64 crossEntropyDerivate(f64 expected, f64 predicted);

#endif