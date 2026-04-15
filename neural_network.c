#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers/types.h"
#include "helpers/mat_calc.h"
#include "functions/activation_functions.h"
#include "functions/cost_functions.h"

#include "neural_network.h"


// Generate a random gaussian number
#define GAUSSIAN_NUM (sqrt(-2.0 * log((rand() + 1.0) / (RAND_MAX + 1.0))) * cos(2.0 * 3.14 * (rand() + 1.0) / (RAND_MAX + 1.0)))

// Coefficent constants for the learning process
#define MOMENTUM_COEF 0.9


Matrix* initializeWeights(u32 currLayer, u32 nextLayer);
Matrix* initializeMomentumW(u32 currLayer, u32 nextLayer);
Matrix* initializeVector(u32 size, u32 zeroOut, u32 extendsRows);
void freeMatrix(Matrix* m);


funcOneParam getFunction(ActivationFunction functionName);
funcOneParam getFunctionDerivate(ActivationFunction functionName);
funcTwoParam getCostFunction(LossFunction functionName);
funcTwoParam getCostFunctionDerivate(LossFunction functionName);


Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName);

void applyActFunction(Layer* layer, u32 size);


void feedForward(Layer* network, u32 numLayers, u32* sizes, f64* input);
void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction, f64** gradWAcc, f64** gradBAcc);
void train(Layer* network, u32 numLayer, u32* sizes, f64** input, f64** expectedOutput, u32 trainSize, LossFunction costFunction, f64 learningRate, u32 epochs, u32 batchSize);


void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes);
void freeNetwork(Layer* network, u32 numLayers);


// Initialize all the weight of a single layer to another with He inizialization
// currLayer and nextLayer are the number of neurons in the layers
// The weights are stored in a matrix and for each index i,j the weight is the one between the i-th neuron of the next layer and the j-th neuron of the current layer
Matrix* initializeWeights(u32 currLayer, u32 nextLayer) {
    Matrix* weights = (Matrix*)malloc(sizeof(Matrix));
    weights->rows = nextLayer;
    weights->cols = currLayer;
    weights->data = (f64*)malloc(currLayer * nextLayer * sizeof(f64));
    f64 std = sqrt(2.0 / currLayer);
    for (u32 i = 0; i < nextLayer; i++) {
        for (u32 j = 0; j < currLayer; j++) {
            SET_MATRIX_ELEMENT(weights, i, j, GAUSSIAN_NUM * std);
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

// Inizialize a Matrix n*1 or 1*n
// Set zeroOut to 1 to iniztialize to zero
// Set extendsRows to 1 to create a n*1 matrix, set it to 0 to create a 1*n matrix
Matrix* initializeVector(u32 size, u32 zeroOut, u32 extendsRows) {
    Matrix* vec = (Matrix*)malloc(sizeof(Matrix));
    vec->rows = extendsRows ? size : 1;
    vec->cols = extendsRows ? 1 : size;
    if (zeroOut) {
        vec->data = (f64*)calloc(size, sizeof(f64));
    } else {
        vec->data = (f64*)malloc(size * sizeof(f64));
    }
    return vec;
}


// Return an activation function pointer based on the enum value
funcOneParam getFunction(ActivationFunction functionName) {
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
funcOneParam getFunctionDerivate(ActivationFunction functionName) {
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

funcTwoParam getCostFunction(LossFunction functionName) {
    switch (functionName){
        case ABSOLUTE_ERROR: return absoluteError; break;
        case SQUARED_ERROR:  return squaredError;  break;
        case LOG_COSH:       return logCosh;       break;
        case CROSS_ENTROPY:  return crossEntropy;  break;
        default:             return NULL;          break;
    }
}

funcTwoParam getCostFunctionDerivate(LossFunction functionName) {
    switch (functionName){
        case ABSOLUTE_ERROR: return absoluteErrorDerivate;   break;
        case SQUARED_ERROR:  return squaredErrorDerivate;    break;
        case LOG_COSH:       return logCoshDerivate;         break;
        case CROSS_ENTROPY:  return crossEntropyDerivate;  break;
        default:             return NULL;                    break;
    }
}

// Initialize the entire neural network given an array of size for each layer, the number of layers and an array of activation function for each layer
Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName) {
    Layer* network = (Layer*)malloc(numLayers * sizeof(Layer));
    
    for (u32 i = 0; i < numLayers; i++) {
        network[i].neurons = initializeVector(sizes[i], 0, 1);
        network[i].zs = i > 0 ? initializeVector(sizes[i], 0, 1) : NULL; // No zs for the input layer
        
        if (i < numLayers - 1) {
            network[i].weights = initializeWeights(sizes[i], sizes[i + 1]);
            network[i].momentumW = initializeMomentumW(sizes[i], sizes[i + 1]);
        } else {
            network[i].weights = NULL; 
            network[i].momentumW = NULL;
        } // No weights nor momentum for the output layer
        
        if (i > 0) {
            network[i].bias = initializeVector(sizes[i], 1, 1);
            network[i].momentumB = initializeVector(sizes[i], 1, 1);
        } else {
            network[i].bias = NULL; 
            network[i].momentumB = NULL;
        } // No bias nor momentum for the input layer
        
        network[i].signalError = i > 0 ? initializeVector(sizes[i], 0, 1) : NULL; // No signal error for the input layer
        network[i].actFunction = getFunction(functionsName[i]);
        network[i].derActFunction = getFunctionDerivate(functionsName[i]);
    }
    
    return network;
}


// Apply the activaction funztio to a layer.
void applyActFunction(Layer* layer, u32 size) {
    funcOneParam layerActFunction = layer->actFunction;
    f64* restrict neuronsData = layer->neurons->data;
    f64* restrict zsData = layer->zs->data;
    for (u32 i = 0; i < size; i++) {
        neuronsData[i] = layerActFunction(zsData[i]);
    }
}


void feedForward(Layer* network, u32 numLayers, u32* sizes, f64* input) {
    // Set the inputs
    for (u32 i = 0; i < sizes[0]; i++) {
        network[0].neurons->data[i] = input[i];
    }
    
    // Calculate the weights * inputs + bias for each layer and apply the activation function
    for (u32 layerIdx = 0; layerIdx < numLayers - 1; layerIdx++) {
        // Calculate the z
        matrixProductWithBias(network[layerIdx].weights, network[layerIdx].neurons, network[layerIdx + 1].bias, network[layerIdx + 1].zs);
        // Apply the activation function
        applyActFunction(&network[layerIdx + 1], sizes[layerIdx + 1]);
    }
}

void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction, f64** gradWAcc, f64** gradBAcc) {
    funcTwoParam cFunc = getCostFunctionDerivate(costFunction);
    u32 outLayer = numLayer - 1;
    for (u32 i = 0; i < sizes[outLayer]; i++) {
        network[outLayer].signalError->data[i] = cFunc(expectedOutput[i], network[outLayer].neurons->data[i]) * network[outLayer].derActFunction(network[outLayer].zs->data[i]);
    }
    
    funcOneParam aFunc; // Activation function derivative
    f64 *signalError, *zs;
    for (u32 layerIdx = outLayer; layerIdx > 0; layerIdx--) {
        if (layerIdx < outLayer) {
            aFunc = network[layerIdx].derActFunction;
            signalError = network[layerIdx].signalError->data;
            zs = network[layerIdx].zs->data;
            for (u32 i = 0; i < sizes[layerIdx]; i++) {
                f64 errorSum = 0;
                for (u32 j = 0; j < sizes[layerIdx + 1]; j++) {
                    errorSum += GET_MATRIX_ELEMENT(network[layerIdx].weights, j, i) * network[layerIdx + 1].signalError->data[j];
                }
                signalError[i] = errorSum * aFunc(zs[i]);
            }
        }
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            gradBAcc[layerIdx - 1][i] += network[layerIdx].signalError->data[i];
            for (u32 j = 0; j < sizes[layerIdx - 1]; j++) {
                gradWAcc[layerIdx - 1][i * sizes[layerIdx - 1] + j] += network[layerIdx].signalError->data[i] * network[layerIdx - 1].neurons->data[j];
            }
        }
    }
}

#define myMin(a, b) ((a) < (b) ? (a) : (b))
// Train the neural network for a certain number of epochs
void train(Layer* network, u32 numLayer, u32* sizes, f64** input, f64** expectedOutput, u32 trainSize, LossFunction costFunction, f64 learningRate, u32 epochs, u32 batchSize) {
    f64** gradWAcc = (f64**)malloc((numLayer - 1) * sizeof(f64*));
    f64** gradBAcc = (f64**)malloc((numLayer - 1) * sizeof(f64*));
    for (u32 l = 0; l < numLayer - 1; l++) {
        gradWAcc[l] = (f64*)malloc(sizes[l + 1] * sizes[l] * sizeof(f64));
        gradBAcc[l] = (f64*)malloc(sizes[l + 1] * sizeof(f64));
    }
    
    f64 avgGradB, avgGradW;
    for (u32 epoch = 0; epoch < epochs; epoch++) {
        for (u32 i = 0; i < trainSize; i += batchSize) {
            u32 currentBatchSize = myMin(trainSize - i, batchSize);
            for (u32 l = 0; l < numLayer - 1; l++) {
                memset(gradWAcc[l], 0, sizes[l + 1] * sizes[l] * sizeof(f64));
                memset(gradBAcc[l], 0, sizes[l + 1] * sizeof(f64));
            }
            for (u32 j = 0; j < currentBatchSize; j++) {
                feedForward(network, numLayer, sizes, input[i + j]);
                backPropagation(network, numLayer, sizes, expectedOutput[i + j], costFunction, gradWAcc, gradBAcc);
            }
            for (u32 l = 0; l < numLayer - 1; l++) {
                u32 nextLayer = l + 1;
                for (u32 r = 0; r < sizes[nextLayer]; r++) {
                    avgGradB = gradBAcc[l][r] / (f64)currentBatchSize;
                    network[nextLayer].momentumB->data[r] = MOMENTUM_COEF * network[nextLayer].momentumB->data[r] + (1.0 - MOMENTUM_COEF) * avgGradB;
                    network[nextLayer].bias->data[r] -= learningRate * network[nextLayer].momentumB->data[r];
                    for (u32 c = 0; c < sizes[l]; c++) {
                        avgGradW = gradWAcc[l][r * sizes[l] + c] / (f64)currentBatchSize;
                        GET_MATRIX_ELEMENT(network[l].momentumW, r, c) = MOMENTUM_COEF * GET_MATRIX_ELEMENT(network[l].momentumW, r, c) + (1.0 - MOMENTUM_COEF) * avgGradW;
                        GET_MATRIX_ELEMENT(network[l].weights, r, c) -= learningRate * GET_MATRIX_ELEMENT(network[l].momentumW, r, c);
                    }
                }
            }
        }
        
        if (!((epoch + 1) % 10) || epoch == 0 || epoch + 1 == epochs) {
            printf("Epoch: %d/%d\n", epoch + 1, epochs);
        }
    }
    
    for (u32 l = 0; l < numLayer - 1; l++) {
        free(gradWAcc[l]);
        free(gradBAcc[l]);
    }
    free(gradWAcc);
    free(gradBAcc);
}


// Print the neurons value of each layer of the neural network
void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes) {
    for (u32 layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            printf("%f ", network[layerIdx].neurons->data[i]);
        }
        printf("\n");
    }
}


void freeMatrix(Matrix* m) {
    if (m != NULL) {
        if (m->data != NULL) free(m->data);
        free(m);
    }
}

void freeNetwork(Layer* network, u32 numLayers) {
    for (u32 i = 0; i < numLayers; i++) {
        freeMatrix(network[i].neurons);
        freeMatrix(network[i].zs);
        freeMatrix(network[i].bias);
        freeMatrix(network[i].momentumB);
        freeMatrix(network[i].signalError);
        freeMatrix(network[i].weights);
        freeMatrix(network[i].momentumW);
    }
    free(network);
}