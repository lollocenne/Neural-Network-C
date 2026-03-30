#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "activation_functions.h"

#define NUM_LAYERS 4
#define SIZES {2, 3, 2, 2}
#define FUNCTIONS {NONE, RELU, SIGMOID, SIGMOID}


typedef f64 (*function)(f64);

typedef enum {
    NONE = -1,
    IDENTITY,
    BINARY_STEP,
    SIGMOID,
    TANH,
    RELU
} ActivationFunction;


f64* initializeNeurons(u32 size);
f64** initializeWeights(u32 currLayer, u32 nextLayer);
f64* initializeBias(u32 size);
function* initializeFuncions(ActivationFunction* functionsName, u32 numLayers);
function* initializeDerivatives(ActivationFunction* functionsName, u32 numLayers);

typedef struct {
    f64* neurons;
    f64** weights;
    f64* bias;
} Layer;

Layer* initializeNetwork(u32* sizes, u32 numLayers);

void freeNetwork(Layer* network, u32* sizes, u32 numLayers);

int main() {
    u32 sizes[NUM_LAYERS] = SIZES;
    ActivationFunction functions[NUM_LAYERS] = FUNCTIONS;
    Layer* model = initializeNetwork(sizes, NUM_LAYERS);
    
    freeNetwork(model, sizes, NUM_LAYERS);
    
    return 0;
}


// Initialize all neurons of a single layer to 0
f64* initializeNeurons(u32 size) {
    return (f64*)calloc(size, sizeof(f64));
}

// Inizialize all the weight of a single layer to another to random values between -1 and 1
// currLayer and nectLayer are the number of neurons in the layers
f64** initializeWeights(u32 currLayer, u32 nextLayer) {
    f64** weights = (f64**)malloc(currLayer * sizeof(f64*));
    for (u32 i = 0; i < currLayer; i++) {
        weights[i] = (f64*)malloc(nextLayer * sizeof(f64));
        for (u32 j = 0; j < nextLayer; j++) {
            weights[i][j] = ((f64)rand() / RAND_MAX) * 2 - 1;
        }
    }
    return weights;
}

// Initialize all bias of a single layer to 0
f64* initializeBias(u32 size) {
    return (f64*)calloc(size, sizeof(f64));
}

// Inizialize the array of function pointers for the activation functions
function* initializeFuncions(ActivationFunction* functionsName, u32 numLayers) {
    function* functions = (function*)malloc(numLayers * sizeof(function));
    
    for (u32 i = 0; i < numLayers; i++) {
        switch (functionsName[i]){
            case IDENTITY:    functions[i] = identity;   break;
            case BINARY_STEP: functions[i] = binaryStep; break;
            case SIGMOID:     functions[i] = sigmoid;    break;
            case TANH:        functions[i] = tanh;       break;
            case RELU:        functions[i] = relu;       break;
            default:          functions[i] = NULL;       break;
        }
    }
    
    return functions;
}

// Inizialize the array of function pointers for the activation functions derivatives
function* initializeDerivatives(ActivationFunction* functionsName, u32 numLayers) {
    function* functions = (function*)malloc(numLayers * sizeof(function));
    
    for (u32 i = 0; i < numLayers; i++) {
        switch (functionsName[i]){
            case IDENTITY:    functions[i] = derivativeIdentity;   break;
            case BINARY_STEP: functions[i] = derivativeBinaryStep; break;
            case SIGMOID:     functions[i] = derivativeSigmoid;    break;
            case TANH:        functions[i] = derivativeTanh;       break;
            case RELU:        functions[i] = derivativeRelu;       break;
            default:          functions[i] = NULL;                 break;
        }
    }
    
    return functions;
}

Layer* initializeNetwork(u32* sizes, u32 numLayers) {
    Layer* network = (Layer*)malloc(numLayers * sizeof(Layer));
    
    for (u32 i = 0; i < numLayers; i++) {
        network[i].neurons = initializeNeurons(sizes[i]);
        if (i < numLayers - 1) {
            network[i].weights = initializeWeights(sizes[i], sizes[i + 1]);
        } else {
            network[i].weights = NULL; // No weights for the output layer
        }
        if (i > 0) {
            network[i].bias = initializeBias(sizes[i]);
        } else {
            network[i].bias = NULL; // No bias for the input layer
        }
    }
    
    return network;
}

void freeNetwork(Layer* network, u32* sizes, u32 numLayers) {
    for (u32 i = 0; i < numLayers; i++) {
        free(network[i].neurons);
        
        if (network[i].bias != NULL) {
            free(network[i].bias);
        }
        
        if (network[i].weights != NULL) {
            for (u32 j = 0; j < sizes[i]; j++) {
                free(network[i].weights[j]);
            }
            free(network[i].weights);
        }
    }
    free(network);
}