#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "helpers/types.h"
#include "helpers/mat_calc.h"
#include "functions/activation_functions.h"
#include "functions/cost_functions.h"

typedef enum {
    NONE = -1,
    IDENTITY,
    BINARY_STEP,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SOFT_PLUS,
    GAUSSIAN,
    SINUSOID
} ActivationFunction;

typedef enum {
    ABSOLUTE_ERROR,
    SQUARED_ERROR,
    LOG_COSH,
    CROSS_ENTROPY
} LossFunction;


Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* actFunctionsName);
void feedForward(Layer* network, u32 numLayers, ActivationFunction* actFunctionsName, u32* sizes, f64* input);
void train(Layer* network, u32 numLayer, u32* sizes, f64** input, f64** expectedOutput, u32 trainSize, ActivationFunction* actFunctionsName, LossFunction costFunction, f64 learningRate, u32 epochs, u32 batchSize);

void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes);
void freeNetwork(Layer* network, u32 numLayers);

#endif