#include <iostream>
#include "neural_network.h"
#include "training/training.h"

int main() {
    std::vector<int> layerSizes = {3, 5, 2};
    NeuralNetwork nn(layerSizes);

    std::vector<std::vector<double>> X_train = {{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}};
    std::vector<std::vector<double>> Y_train = {{1.0, 0.0}, {0.0, 1.0}};

    train(nn, X_train, Y_train, 1000, 0.01);

    return 0;
}
