#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "../types.h"

f64 squaredError(f64 expected, f64 predicted);

f64 squaredErrorDerivate(f64 expected, f64 predicted);

#endif