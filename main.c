#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "functions/activation_functions.h"
#include "functions/cost_functions.h"

#define NUM_LAYERS 2
#define SIZES {1, 1}
#define FUNCTIONS {NONE, IDENTITY}

// Should predict double + 1
#define TRAIN_SIZE 3
#define INPUTS {(f64[]){1.0}, (f64[]){4.0}, (f64[]){-3.0}}
#define EXP_OUTPUT {(f64[]){3.0}, (f64[]){9.0}, (f64[]){-5.0}}


#define LEARNING_RATE 0.1
#define MOMENTUM_COEF 0.9


typedef f64 (*Actfunction)(f64);
typedef f64 (*Cstfunction)(f64, f64);

typedef enum {
    NONE = -1,
    IDENTITY,
    BINARY_STEP,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU
} ActivationFunction;

typedef enum {
    ABSOLUTE_ERROR,
    SQUARED_ERROR,
    LOG_COSH
} LossFunction;


typedef struct {
    u32 rows;
    u32 cols;
    f64* data;
} Matrix;

#define GET_MATRIX_ELEMENT(matrix, row, col) ((matrix)->data[(row) * (matrix)->cols + (col)])
#define SET_MATRIX_ELEMENT(matrix, row, col, val) ((matrix)->data[(row) * (matrix)->cols + (col)] = (val))


Matrix* initializeWeights(u32 currLayer, u32 nextLayer);
Matrix* initializeMomentumW(u32 currLayer, u32 nextLayer);
Actfunction getFunction(ActivationFunction functionName);
Actfunction getFunctionDerivate(ActivationFunction functionName);
Cstfunction getCostFunction(LossFunction functionName);
Cstfunction getCostFunctionDerivate(LossFunction functionName);

typedef struct {
    f64* neurons;
    f64* zs; // the value before applying the activation function
    Matrix* weights;
    f64* bias;
    Matrix* momentumW;
    f64* momentumB;
    f64* signalError; // the error signal for each neuron
    Actfunction actFunction;
    Actfunction derActFunction;
} Layer;

Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName);

void feedForward(Layer* network, u32 numLayers, u32* sizes, f64* input);
void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction, f64 learningRate);
void learn(Layer* network, u32 numLayer, u32* sizes, f64* input, f64* expectedOutput, LossFunction costFunction, f64 learningRate);
void train(Layer* network, u32 numLayer, u32* sizes, f64** input, f64** expectedOutput, u32 trainSize, LossFunction costFunction, f64 learningRate, u32 epochs);


void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes);
void freeNetwork(Layer* network, u32 numLayers);


int main() {    
    u32 sizes[NUM_LAYERS] = SIZES;
    ActivationFunction functions[NUM_LAYERS] = FUNCTIONS;
    Layer* model = initializeNetwork(sizes, NUM_LAYERS, functions);
    f64* inputs[] = INPUTS;
    f64* expectedOutput[] = EXP_OUTPUT;
    train(model, NUM_LAYERS, sizes, inputs, expectedOutput, TRAIN_SIZE, SQUARED_ERROR, LEARNING_RATE, 100);
    feedForward(model, NUM_LAYERS, sizes, inputs[0]);
    printf("Predicted: %f\nExpected: %f\n", model[1].neurons[0], expectedOutput[0][0]);
    printf("weight: %f     bias: %f\n", model[0].weights->data[0], model[1].bias[0]);
    
    return 0;
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

// Inizialize all the momentum of the weights of a single layer
Matrix* initializeMomentumW(u32 currLayer, u32 nextLayer) {
    Matrix* weights = (Matrix*)malloc(sizeof(Matrix));
    weights->rows = nextLayer;
    weights->cols = currLayer;
    weights->data = (f64*)calloc(currLayer * nextLayer, sizeof(f64));
    return weights;
}

// Return an activation function pointer based on the enum value
Actfunction getFunction(ActivationFunction functionName) {
    switch (functionName){
        case IDENTITY:    return identity;   break;
        case BINARY_STEP: return binaryStep; break;
        case SIGMOID:     return sigmoid;    break;
        case TANH:        return tanh;       break;
        case RELU:        return relu;       break;
        case LEAKY_RELU:  return leakyRelu;  break;
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
        case LEAKY_RELU:  return derivativeLeakyRelu;  break;
        default:          return NULL;                 break;
    }
}

Cstfunction getCostFunction(LossFunction functionName) {
    switch (functionName){
        case ABSOLUTE_ERROR: return absoluteError; break;
        case SQUARED_ERROR:  return squaredError;  break;
        case LOG_COSH:       return logCosh;       break;
        default:             return NULL;          break;
    }
}

Cstfunction getCostFunctionDerivate(LossFunction functionName) {
    switch (functionName){
        case ABSOLUTE_ERROR: return squaredErrorDerivate; break;
        case SQUARED_ERROR:  return squaredErrorDerivate; break;
        case LOG_COSH:       return logCoshDerivate;      break;
        default:             return NULL;                 break;
    }
}

// Initialize the entire neural network given an array of size for each layer, the number of layers and an array of activation function for each layer
Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName) {
    Layer* network = (Layer*)malloc(numLayers * sizeof(Layer));
    
    for (u32 i = 0; i < numLayers; i++) {
        network[i].neurons = (f64*)malloc(sizes[i] * sizeof(f64));
        network[i].zs = i > 0 ? (f64*)malloc(sizes[i] * sizeof(f64)) : NULL; // No zs for the input layer
        
        if (i < numLayers - 1) {
            network[i].weights = initializeWeights(sizes[i], sizes[i + 1]);
            network[i].momentumW = initializeMomentumW(sizes[i], sizes[i + 1]);
        } else {network[i].weights = NULL; network[i].momentumW = NULL;} // No weights noe momentum for the output layer
        
        if (i > 0) {
            network[i].bias = (f64*)calloc(sizes[i], sizeof(f64));
            network[i].momentumB = (f64*)calloc(sizes[i], sizeof(f64));;
        } else {network[i].bias = NULL; network[i].momentumB = NULL;} // No bias nor momentum for the input layer
        
        network[i].signalError = i > 0 ? (f64*)malloc(sizes[i] * sizeof(f64)) : NULL; // No signal error for the input layer
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
            network[layerIdx + 1].zs[i] = network[layerIdx + 1].bias[i];
            for (u32 j = 0; j < sizes[layerIdx]; j++) {
                network[layerIdx + 1].zs[i] += network[layerIdx].neurons[j] * GET_MATRIX_ELEMENT(network[layerIdx].weights, i, j);
            }
        }
    }
    
    // apply the activation function to each neuron
    for (u32 layerIdx = 1; layerIdx < numLayers; layerIdx++) {
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            network[layerIdx].neurons[i] = network[layerIdx].actFunction(network[layerIdx].zs[i]);
        }
    }
}

void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction, f64 learningRate) {
    // Update the weights and bias of the output layer
    for (u32 i = 0; i < sizes[numLayer - 1]; i++) {
        network[numLayer - 1].signalError[i] = getCostFunctionDerivate(costFunction)(expectedOutput[i], network[numLayer - 1].neurons[i]) * network[numLayer - 1].derActFunction(network[numLayer - 1].zs[i]);
        for (u32 j = 0; j < sizes[numLayer - 2]; j++) {
            GET_MATRIX_ELEMENT(network[numLayer - 2].momentumW, i, j) = MOMENTUM_COEF * GET_MATRIX_ELEMENT(network[numLayer - 2].momentumW, i, j) + (1 - MOMENTUM_COEF) * network[numLayer - 1].signalError[i] * network[numLayer - 2].neurons[j];
            GET_MATRIX_ELEMENT(network[numLayer - 2].weights, i, j) -= learningRate * GET_MATRIX_ELEMENT(network[numLayer - 2].momentumW, i, j);
        }
        network[numLayer - 1].momentumB[i] = MOMENTUM_COEF * network[numLayer - 1].momentumB[i] + (1 - MOMENTUM_COEF) * network[numLayer - 1].signalError[i];
        network[numLayer - 1].bias[i] -= learningRate * network[numLayer - 1].momentumB[i];
    }
    
    // Update the weights and bias of all the the layer
    for (u32 layerIdx = numLayer - 2; layerIdx > 0; layerIdx--) {
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            network[layerIdx].signalError[i] = 0;
            for (u32 j = 0; j < sizes[layerIdx + 1]; j++) {
                network[layerIdx].signalError[i] += GET_MATRIX_ELEMENT(network[layerIdx].weights, j, i) * network[layerIdx + 1].signalError[j];
            }
            network[layerIdx].signalError[i] *= network[layerIdx].derActFunction(network[layerIdx].zs[i]);
            for (u32 j = 0; j < sizes[layerIdx - 1]; j++) {
                GET_MATRIX_ELEMENT(network[layerIdx - 1].momentumW, i, j) = MOMENTUM_COEF * GET_MATRIX_ELEMENT(network[layerIdx - 1].momentumW, i, j) + (1 - MOMENTUM_COEF) * network[layerIdx].signalError[i] * network[layerIdx - 1].neurons[j];
                GET_MATRIX_ELEMENT(network[layerIdx - 1].weights, i, j) -= learningRate * GET_MATRIX_ELEMENT(network[layerIdx - 1].momentumW, i, j);
            }
            network[layerIdx].momentumB[i] = MOMENTUM_COEF * network[layerIdx].momentumB[i] + (1 - MOMENTUM_COEF) * network[layerIdx].signalError[i];
            network[layerIdx].bias[i] -= learningRate * network[layerIdx].momentumB[i];
        }
    }
}

// Train the neural network for a single input and expected output
void learn(Layer* network, u32 numLayer, u32* sizes, f64* input, f64* expectedOutput, LossFunction costFunction, f64 learningRate) {
    feedForward(network, numLayer, sizes, input);
    backPropagation(network, numLayer, sizes, expectedOutput, costFunction, learningRate);
}

// Train the neural network for a certain number of epochs
void train(Layer* network, u32 numLayer, u32* sizes, f64** input, f64** expectedOutput, u32 trainSize, LossFunction costFunction, f64 learningRate, u32 epochs) {
    for (u32 epoch = 0; epoch < epochs; epoch++) {
        for (u32 i = 0; i < trainSize; i++) {
            learn(network, numLayer, sizes, input[i], expectedOutput[i], costFunction, learningRate);
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
        if (network[i].neurons     != NULL) free(network[i].neurons);
        if (network[i].zs          != NULL) free(network[i].zs);
        if (network[i].bias        != NULL) free(network[i].bias);
        if (network[i].momentumB   != NULL) free(network[i].momentumB);
        if (network[i].signalError != NULL) free(network[i].signalError);
        
        if (network[i].weights     != NULL) {
            free(network[i].weights->data);
            free(network[i].weights);
        }
        if (network[i].momentumW   != NULL) {
            free(network[i].momentumW->data);
            free(network[i].momentumW);
        }
    }
    free(network);
}