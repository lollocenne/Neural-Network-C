#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "activation_functions.h"
#include "cost_functions.h"

#define NUM_LAYERS 4
#define LAYERS_SIZES {2, 3, 2, 2}
#define FUNCTIONS {NONE, RELU, SIGMOID, SIGMOID}
#define INPUTS {0.5, 0.7}


typedef f64 (*Actfunction)(f64);
typedef f64 (*Costfunction)(f64, f64);

typedef enum {
    NONE = -1,
    IDENTITY,
    BINARY_STEP,
    SIGMOID,
    TANH,
    RELU
} ActivationFunction;

typedef enum {
    SQUARED_ERROR
} LossFunction;


typedef struct {
    u32 rows;
    u32 cols;
    f64* data;
} Matrix;

#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))


f64* initializeNeurons(u32 size);
Matrix* initializeWeights(u32 currLayer, u32 nextLayer);
f64* initializeBias(u32 size);
Actfunction getFunction(ActivationFunction functionName);
Actfunction getFunctionDerivate(ActivationFunction functionName);

typedef struct {
    f64* neurons;
    f64* zs; // the value before applying the activation function
    Matrix* weights;
    f64* bias;
    Actfunction actFunction;
    Actfunction derActFunction;
} Layer;

Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName);

void feedForward(Layer* network, u32 numLayers, u32* sizes, f64* input);

void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction);


void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes);
void freeNetwork(Layer* network, u32 numLayers);


int main() {
    u32 sizes[NUM_LAYERS] = LAYERS_SIZES;
    ActivationFunction functions[NUM_LAYERS] = FUNCTIONS;
    Layer* model = initializeNetwork(sizes, NUM_LAYERS, functions);
    f64 inputs[2] = INPUTS;
    
    printNeuralNetwork(model, NUM_LAYERS, sizes);
    feedForward(model, NUM_LAYERS, sizes, inputs);
    printf("\n-------------------------------------\n");
    printNeuralNetwork(model, NUM_LAYERS, sizes);
    
    freeNetwork(model, NUM_LAYERS);
    
    return 0;
}


// Initialize all neurons of a single layer to 0
f64* initializeNeurons(u32 size) {
    return (f64*)calloc(size, sizeof(f64)); //TODO: use malloc, no need to be 0
}

// Initialize all the weight of a single layer to another to random values between -1 and 1
// currLayer and nextLayer are the number of neurons in the layers
// The weights are stored in a matrix and for each index i,j the weight is the one between the i-th neuron of the next layer and the j-th neuron of the current layer
Matrix* initializeWeights(u32 currLayer, u32 nextLayer) {
    Matrix* weights = (Matrix*)malloc(sizeof(Matrix));
    weights->rows = nextLayer;
    weights->cols = currLayer;
    weights->data = (f64*)malloc(currLayer * nextLayer * sizeof(f64));
    for (u32 i = 0; i < nextLayer; i++) {
        for (u32 j = 0; j < currLayer; j++) {
            SET_MATRIX_ELEMENT(weights, i, j, ((f64)rand() / RAND_MAX) * 2 - 1);
        }
    }
    return weights;
}

// Initialize all bias of a single layer to 0
f64* initializeBias(u32 size) {
    return (f64*)calloc(size, sizeof(f64));
}

// Return an activation function pointer based on the enum value
Actfunction getFunction(ActivationFunction functionName) {
    switch (functionName){
        case IDENTITY:    return identity;   break;
        case BINARY_STEP: return binaryStep; break;
        case SIGMOID:     return sigmoid;    break;
        case TANH:        return tanh;       break;
        case RELU:        return relu;       break;
        default:          return NULL;       break;
    }
}

// Return an activation function derivative pointer based on the enum value
Actfunction getFunctionDerivate(ActivationFunction functionName) {
    switch (functionName){
        case IDENTITY:    return derivativeIdentity;   break;
        case BINARY_STEP: return derivativeBinaryStep; break;
        case SIGMOID:     return derivativeSigmoid;    break;
        case TANH:        return derivativeTanh;       break;
        case RELU:        return derivativeRelu;       break;
        default:          return NULL;                 break;
    }
}

// Initialize the entire neural network given an array of size for each layer, the number of layers and an array of activation function for each layer
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

void feedForward(Layer* network, u32 numLayers, u32* sizes, f64* input) {
    // Set the inputs
    for (u32 i = 0; i < sizes[0]; i++) {
        network[0].neurons[i] = input[i];
    }
    
    // Calculate the weights * inputs + bias for each layer
    for (u32 layerIdx = 0; layerIdx < numLayers - 1; layerIdx++) {        
        for (u32 i = 0; i < sizes[layerIdx + 1]; i++) {
            network[layerIdx + 1].neurons[i] = network[layerIdx + 1].bias[i];
            for (u32 j = 0; j < sizes[layerIdx]; j++) {
                network[layerIdx + 1].neurons[i] += network[layerIdx].neurons[j] * GET_MATRIX_ELEMENT(network[layerIdx].weights, i, j);
            }
        }
    }
    
    // apply the activation function to each neuron
    for (u32 layerIdx = 1; layerIdx < numLayers; layerIdx++) {
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            network[layerIdx].zs[i] = network[layerIdx].neurons[i];
            network[layerIdx].neurons[i] = network[layerIdx].actFunction(network[layerIdx].neurons[i]);
        }
    }
}

// Print the neurons value of each layer of the neural network
void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes) {
    for (u32 layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            printf("%f ", network[layerIdx].neurons[i]);
        }
        printf("\n");
    }
}

void freeNetwork(Layer* network, u32 numLayers) {
    for (u32 i = 0; i < numLayers; i++) {
        if (network[i].neurons != NULL) {
            free(network[i].neurons);
        }
        
        if (network[i].bias != NULL) {
            free(network[i].bias);
        }
        
        if (network[i].weights != NULL) {
            free(network[i].weights->data);
            free(network[i].weights);
        }
    }
    free(network);
}