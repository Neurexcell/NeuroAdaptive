#include "neural_network.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& architecture) {
    assert(architecture.size() > 1);

    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        layers.push_back(Layer{
            std::vector<std::vector<double>>(architecture[i + 1], std::vector<double>(architecture[i], 0.0)),
            std::vector<std::vector<double>>(architecture[i + 1], std::vector<double>(architecture[i], 0.1)),
            std::vector<double>(architecture[i + 1], 0.0)
        });
    }

    initialize_weights();
}

void NeuralNetwork::initialize_weights() {
    for (auto& layer : layers) {
        for (size_t i = 0; i < layer.neurons.size(); ++i) {
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                layer.weights[i][j] = (rand() % 100) / 1000.0;
            }
            layer.biases[i] = 0.0;
        }
    }
}

double NeuralNetwork::relu(double x) {
    return x > 0 ? x : 0;
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> input_data = input;
    for (auto& layer : layers) {
        std::vector<double> output(layer.neurons.size(), 0.0);
        for (size_t i = 0; i < layer.neurons.size(); ++i) {
            for (size_t j = 0; j < input_data.size(); ++j) {
                output[i] += input_data[j] * layer.weights[i][j];
            }
            output[i] += layer.biases[i];
            layer.neurons[i] = relu(output[i]);
        }
        input_data = output;
    }
    return input_data;
}

