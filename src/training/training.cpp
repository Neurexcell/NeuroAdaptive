#include "training.h"
#include <iostream>
#include <cmath>

void train(NeuralNetwork &nn, const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        for (size_t i = 0; i < X_train.size(); ++i) {
            nn.forward(X_train[i]);
            for (size_t j = 0; j < Y_train[i].size(); ++j) {
                double error = Y_train[i][j] - nn.getActivation(j);
                total_error += error * error;
            }
            nn.backward(Y_train[i]);
            nn.updateWeights(learning_rate);
        }
        std::cout << "Epoch " << epoch + 1 << " - Error: " << total_error / X_train.size() << std::endl;
    }
}

void test(NeuralNetwork &nn, const std::vector<std::vector<double>> &X_test, const std::vector<std::vector<double>> &Y_test) {
    double total_error = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        nn.forward(X_test[i]);
        for (size_t j = 0; j < Y_test[i].size(); ++j) {
            double error = Y_test[i][j] - nn.getActivation(j);
            total_error += error * error;
        }
    }
    std::cout << "Test Error: " << total_error / X_test.size() << std::endl;
}

