#ifndef TRAINING_H
#define TRAINING_H

#include <vector>
#include "neural_network.h"

void train(NeuralNetwork &nn, const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, int epochs, double learning_rate);
void test(NeuralNetwork &nn, const std::vector<std::vector<double>> &X_test, const std::vector<std::vector<double>> &Y_test);

#endif

