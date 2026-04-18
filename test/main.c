#include <stdio.h>
#include <string.h>
#include <time.h>
#include "../helpers/types.h"
#include "../helpers/mat_calc.h"
#include "../functions/activation_functions.h"
#include "../functions/cost_functions.h"
#include "../neural_network.h"


// Example and test constant
#define NUM_LAYERS 4
#define SIZES {784, 128, 64, 10}
#define FUNCTIONS {NONE, LEAKY_RELU, LEAKY_RELU, SIGMOID}
#define LOSS_FUNCTION CROSS_ENTROPY


// Coefficent constants for the learning process
#define LEARNING_RATE 0.1


u32 **parseCsv(const char *filename, u32 *rowCount);
u32 createDataset(const char *filename, f64 ***inputs, f64 ***expectedOutput, u32 limit);


int main() {
    // Testing Time
    clock_t start;
    start = clock();
    
    
    srand(time(NULL));
    
    u32 sizes[NUM_LAYERS] = SIZES;
    ActivationFunction functions[NUM_LAYERS] = FUNCTIONS;
    
    Layer* model = initializeNetwork(sizes, NUM_LAYERS, functions);
    
    f64** inputs;
    f64** expectedOutput;
    u32 datasetSize = createDataset("test/mnist_test.csv", &inputs, &expectedOutput, 5000);
    const u32 trainSize = datasetSize * 0.8;
    const u32 testSize = datasetSize * 0.2;
    
    printf("Training...\n");
    train(model, NUM_LAYERS, sizes, inputs, expectedOutput, trainSize, functions, LOSS_FUNCTION, LEARNING_RATE, 64, 32);
    
    // Accuracy test
    printf("Testing...\n");
    f64** inputsTest = inputs + trainSize;
    f64** outputsTest = expectedOutput + trainSize;
    u32 correct = 0;
    for (u32 i = 0; i < testSize; i++) {
        feedForward(model, NUM_LAYERS, functions, sizes, inputsTest[i]);
        
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
    
    
    // Testing Time
    printf("Seconds: %f\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
    
    return 0;
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