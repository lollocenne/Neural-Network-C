#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef int32_t  i32;

typedef uint32_t u32;

typedef float    f32;
typedef double   f64;


typedef f64 (*funcOneParam)(f64);
typedef f64 (*funcTwoParam)(f64, f64);

#endif