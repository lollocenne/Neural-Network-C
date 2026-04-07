#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "helpers/types.h"
#include "helpers/mat_calc.h"
#include "functions/activation_functions.h"
#include "functions/cost_functions.h"


// Example and test constant
#define NUM_LAYERS 4
#define SIZES {784, 128, 64, 10}
#define FUNCTIONS {NONE, LEAKY_RELU, LEAKY_RELU, SIGMOID}
#define LOSS_FUNCTION SQUARED_ERROR


// Generate a random gaussian number
#define GAUSSIAN_NUM (sqrt(-2.0 * log((rand() + 1.0) / (RAND_MAX + 1.0))) * cos(2.0 * 3.14 * (rand() + 1.0) / (RAND_MAX + 1.0)))

// Coefficent constants for the learning process
#define LEARNING_RATE 0.01
#define MOMENTUM_COEF 0.9


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
    LOG_COSH,
    CROSS_ENTROPY
} LossFunction;


Matrix* initializeWeights(u32 currLayer, u32 nextLayer);
Matrix* initializeMomentumW(u32 currLayer, u32 nextLayer);
Matrix* initializeVector(u32 size, u32 zeroOut);
void freeMatrix(Matrix* m);


funcOneParam getFunction(ActivationFunction functionName);
funcOneParam getFunctionDerivate(ActivationFunction functionName);
funcTwoParam getCostFunction(LossFunction functionName);
funcTwoParam getCostFunctionDerivate(LossFunction functionName);

typedef struct {
    Matrix* neurons;
    Matrix* zs; // the value before applying the activation function
    Matrix* weights;
    Matrix* bias;
    Matrix* momentumW;
    Matrix* momentumB;
    Matrix* signalError; // the error signal for each neuron
    funcOneParam actFunction;
    funcOneParam derActFunction;
} Layer;

Layer* initializeNetwork(u32* sizes, u32 numLayers, ActivationFunction* functionsName);

void applyActFunction(Layer* layer, u32 size);

void feedForward(Layer* network, u32 numLayers, u32* sizes, f64* input);
void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction, f64 learningRate);
void learn(Layer* network, u32 numLayer, u32* sizes, f64* input, f64* expectedOutput, LossFunction costFunction, f64 learningRate);
void train(Layer* network, u32 numLayer, u32* sizes, f64** input, f64** expectedOutput, u32 trainSize, LossFunction costFunction, f64 learningRate, u32 epochs);


void printNeuralNetwork(Layer* network, u32 numLayers, u32* sizes);
void freeNetwork(Layer* network, u32 numLayers);


u32 **parseCsv(const char *filename, u32 *rowCount);
u32 createDataset(const char *filename, f64 ***inputs, f64 ***expectedOutput, u32 limit);


int main() {    
    srand(time(NULL));
    
    u32 sizes[NUM_LAYERS] = SIZES;
    ActivationFunction functions[NUM_LAYERS] = FUNCTIONS;
    
    Layer* model = initializeNetwork(sizes, NUM_LAYERS, functions);
    
    f64** inputs;
    f64** expectedOutput;
    u32 datasetSize = createDataset("mnist_test.csv", &inputs, &expectedOutput, 5000);
    const u32 trainSize = datasetSize * 0.8;
    const u32 testSize = datasetSize * 0.2;
    
    printf("Training...\n");
    train(model, NUM_LAYERS, sizes, inputs, expectedOutput, trainSize, LOSS_FUNCTION, LEARNING_RATE, 128);
    
    // Accuracy test
    printf("Testing...\n");
    f64** inputsTest = inputs + trainSize;
    f64** outputsTest = expectedOutput + trainSize;
    u32 correct = 0;
    for (u32 i = 0; i < testSize; i++) {
        feedForward(model, NUM_LAYERS, sizes, inputsTest[i]);
        
        u32 predicted = 0, expected = 0;
        for (u32 j = 0; j < 10; j++) {
            if (model[NUM_LAYERS - 1].neurons->data[j] > model[NUM_LAYERS - 1].neurons->data[predicted]) {
                predicted = j;
            }
        }
        for (u32 j = 0; j < 10; j++) {
            if (outputsTest[i][j] > outputsTest[i][expected]) {
                expected = j;
            }
        }
        if (predicted == expected) correct++;
    }
    printf("Accuracy: %.2f%%\n", 100.0 * correct / testSize);
    
    for (u32 i = 0; i < datasetSize; i++) {
        free(inputs[i]);
        free(expectedOutput[i]);
    }
    free(inputs);
    free(expectedOutput);
    freeNetwork(model, NUM_LAYERS);
    
    return 0;
}


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

Matrix* initializeVector(u32 size, u32 zeroOut) {
    Matrix* vec = (Matrix*)malloc(sizeof(Matrix));
    vec->rows = size;
    vec->cols = 1;
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
        network[i].neurons = initializeVector(sizes[i], 0);
        network[i].zs = i > 0 ? initializeVector(sizes[i], 0) : NULL; // No zs for the input layer
        
        if (i < numLayers - 1) {
            network[i].weights = initializeWeights(sizes[i], sizes[i + 1]);
            network[i].momentumW = initializeMomentumW(sizes[i], sizes[i + 1]);
        } else {
            network[i].weights = NULL; 
            network[i].momentumW = NULL;
        } // No weights nor momentum for the output layer
        
        if (i > 0) {
            network[i].bias = initializeVector(sizes[i], 1);
            network[i].momentumB = initializeVector(sizes[i], 1);
        } else {
            network[i].bias = NULL; 
            network[i].momentumB = NULL;
        } // No bias nor momentum for the input layer
        
        network[i].signalError = i > 0 ? initializeVector(sizes[i], 0) : NULL; // No signal error for the input layer
        network[i].actFunction = getFunction(functionsName[i]);
        network[i].derActFunction = getFunctionDerivate(functionsName[i]);
    }
    
    return network;
}


// Apply the activaction funztio to a layer.
void applyActFunction(Layer* layer, u32 size) {
    for (u32 i = 0; i < size; i++) {
        layer->neurons->data[i] = layer->actFunction(layer->zs->data[i]);
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
        matrixProduct(network[layerIdx].weights, network[layerIdx].neurons, network[layerIdx + 1].zs);
        sumMatrices(network[layerIdx + 1].zs, network[layerIdx + 1].bias, network[layerIdx + 1].zs);
        // Apply the activation function
        applyActFunction(&network[layerIdx + 1], sizes[layerIdx + 1]);
    }
}

void backPropagation(Layer* network, u32 numLayer, u32* sizes, f64* expectedOutput, LossFunction costFunction, f64 learningRate) {
    f64 gradient;
    // Update the weights and bias of the output layer
    for (u32 i = 0; i < sizes[numLayer - 1]; i++) {
        network[numLayer - 1].signalError->data[i] = getCostFunctionDerivate(costFunction)(expectedOutput[i], network[numLayer - 1].neurons->data[i]) * network[numLayer - 1].derActFunction(network[numLayer - 1].zs->data[i]);
        for (u32 j = 0; j < sizes[numLayer - 2]; j++) {
            gradient = network[numLayer - 1].signalError->data[i] * network[numLayer - 2].neurons->data[j];
            GET_MATRIX_ELEMENT(network[numLayer - 2].momentumW, i, j) = MOMENTUM_COEF * GET_MATRIX_ELEMENT(network[numLayer - 2].momentumW, i, j) + (1.0 - MOMENTUM_COEF) * gradient;
            GET_MATRIX_ELEMENT(network[numLayer - 2].weights, i, j) -= learningRate * GET_MATRIX_ELEMENT(network[numLayer - 2].momentumW, i, j);
        }
        network[numLayer - 1].momentumB->data[i] = MOMENTUM_COEF * network[numLayer - 1].momentumB->data[i] + (1.0 - MOMENTUM_COEF) * network[numLayer - 1].signalError->data[i];
        network[numLayer - 1].bias->data[i] -= learningRate * network[numLayer - 1].momentumB->data[i];
    }
    
    // Update the weights and bias of all the the layer
    for (u32 layerIdx = numLayer - 2; layerIdx > 0; layerIdx--) {
        for (u32 i = 0; i < sizes[layerIdx]; i++) {
            network[layerIdx].signalError->data[i] = 0;
            for (u32 j = 0; j < sizes[layerIdx + 1]; j++) {
                network[layerIdx].signalError->data[i] += GET_MATRIX_ELEMENT(network[layerIdx].weights, j, i) * network[layerIdx + 1].signalError->data[j];
            }
            network[layerIdx].signalError->data[i] *= network[layerIdx].derActFunction(network[layerIdx].zs->data[i]);
            for (u32 j = 0; j < sizes[layerIdx - 1]; j++) {
                gradient = network[layerIdx].signalError->data[i] * network[layerIdx - 1].neurons->data[j];
                GET_MATRIX_ELEMENT(network[layerIdx - 1].momentumW, i, j) = MOMENTUM_COEF * GET_MATRIX_ELEMENT(network[layerIdx - 1].momentumW, i, j) + (1.0 - MOMENTUM_COEF) * gradient;
                GET_MATRIX_ELEMENT(network[layerIdx - 1].weights, i, j) -= learningRate * GET_MATRIX_ELEMENT(network[layerIdx - 1].momentumW, i, j);
            }
            network[layerIdx].momentumB->data[i] = MOMENTUM_COEF * network[layerIdx].momentumB->data[i] + (1.0 - MOMENTUM_COEF) * network[layerIdx].signalError->data[i];
            network[layerIdx].bias->data[i] -= learningRate * network[layerIdx].momentumB->data[i];
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
        if (!((epoch + 1) % 10) || epoch == 0 || epoch + 1 == epochs) printf("Epoch: %d/%d\n", epoch + 1, epochs);
    }
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


// Load the data from mnist csv file
#define COLS 785 // 1 label + 784 pixels
u32 **parseCsv(const char *filename, u32 *rowCount) {
    FILE *f = fopen(filename, "r");
    if (!f) { perror("fopen"); return NULL; }
    
    char line[COLS * 6 + COLS];
    fgets(line, sizeof(line), f);
    
    u32 capacity = 1000;
    *rowCount = 0;
    u32 **data = malloc(capacity * sizeof(u32 *));
    
    while (fgets(line, sizeof(line), f)) {
        if (*rowCount == capacity) {
            capacity *= 2;
            data = realloc(data, capacity * sizeof(u32 *));
        }
        
        u32 *row = malloc(COLS * sizeof(u32));
        char *token = strtok(line, ",\n");
        u32 col = 0;
        
        while (token && col < COLS) {
            row[col++] = atoi(token);
            token = strtok(NULL, ",\n");
        }
        
        if (col == COLS)
            data[(*rowCount)++] = row;
        else
            free(row);
    }
    
    fclose(f);
    return data;
}

#define PIXELS 784
#define CLASSES 10

u32 createDataset(const char *filename, f64 ***inputs, f64 ***expectedOutput, u32 limit) {
    u32 totalRows;
    u32 **data = parseCsv(filename, &totalRows);
    
    u32 rowCount = limit < totalRows ? limit : totalRows;
    
    *inputs = malloc(rowCount * sizeof(f64 *));
    *expectedOutput = malloc(rowCount * sizeof(f64 *));
    
    for (u32 i = 0; i < totalRows; i++) {
        if (i < rowCount) {
            f64 *input = malloc(PIXELS * sizeof(f64));
            for (u32 j = 0; j < PIXELS; j++)
                input[j] = data[i][j + 1] / 255.0;
            (*inputs)[i] = input;
            
            f64 *label = calloc(CLASSES, sizeof(f64));
            label[data[i][0]] = 1.0;
            (*expectedOutput)[i] = label;
        }
        free(data[i]);
    }
    
    free(data);
    return rowCount;
}