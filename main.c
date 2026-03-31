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
function getFunction(ActivationFunction functionName);
function getFunctionDerivate(ActivationFunction functionName);

typedef struct {
    f64* neurons;
    f64** weights;
    f64* bias;
    function actFunction;
    function derActFunction;
} Layer;

Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName);

void freeNetwork(Layer* network, u32* sizes, u32 numLayers);

int main() {
    u32 sizes[NUM_LAYERS] = SIZES;
    ActivationFunction functions[NUM_LAYERS] = FUNCTIONS;
    Layer* model = initializeNetwork(sizes, NUM_LAYERS, functions);
    
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

// Return an activation function pointer based on the enum value
function getFunction(ActivationFunction functionName) {
    switch (functionName){
        case IDENTITY:    return identity;   break;
        case BINARY_STEP: return binaryStep; break;
        case SIGMOID:     return sigmoid;    break;
        case TANH:        return tanh;       break;
        case RELU:        return relu;       break;
        default:          return NULL;       break;
    }
    return NULL;
}

// Return an activation function derivative pointer based on the enum value
function getFunctionDerivate(ActivationFunction functionName) {
    switch (functionName){
        case IDENTITY:    return derivativeIdentity;   break;
        case BINARY_STEP: return derivativeBinaryStep; break;
        case SIGMOID:     return derivativeSigmoid;    break;
        case TANH:        return derivativeTanh;       break;
        case RELU:        return derivativeRelu;       break;
        default:          return NULL;                 break;
    }
    return NULL;
}

Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName) {
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
        network[i].actFunction = getFunction(functionsName[i]);
        network[i].derActFunction = getFunctionDerivate(functionsName[i]);
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