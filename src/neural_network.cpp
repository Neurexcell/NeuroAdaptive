#include "neural_network.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& architecture) {
    assert(architecture.size() > 1);  // Ensure at least 1 hidden layer

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
                layer.weights[i][j] = (rand() % 100) / 1000.0; // Random small values
            }
            layer.biases[i] = 0.0;
        }
    }
}

double NeuralNetwork::relu(double x) {
    return x > 0 ? x : 0;
}

double NeuralNetwork::relu_derivative(double x) {
    return x > 0 ? 1 : 0;  // Derivative of ReLU
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));  // Sigmoid activation function
}

double NeuralNetwork::sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));  // Sigmoid derivative
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

void NeuralNetwork::train(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < data.size(); ++i) {
            // Forward pass
            std::vector<double> output = forward(data[i]);

            // Calculate error (Mean Squared Error)
            double error = calculate_error(output, labels[i]);
            std::cout << "Epoch " << epoch << " - Error: " << error << std::endl;

            // Backward pass (Backpropagation)
            std::vector<std::vector<double>> output_deltas;
            std::vector<std::vector<double>> input_activations;

            for (auto& layer : layers) {
                output_deltas.push_back(std::vector<double>(layer.neurons.size(), 0.0));
                input_activations.push_back(std::vector<double>(layer.neurons.size(), 0.0));
            }

            // You will need to implement the backpropagation step here to calculate
            // the gradients, and adjust weights/biases accordingly
            update_weights(output_deltas, input_activations, learning_rate);
        }
    }
}

double NeuralNetwork::calculate_error(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double error = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        error += pow(predicted[i] - actual[i], 2);  // Mean Squared Error
    }
    return error / predicted.size();
}

void NeuralNetwork::update_weights(const std::vector<std::vector<double>>& output_deltas,
                                    const std::vector<std::vector<double>>& input_activations,
                                    double learning_rate) {
    // Implement weight update using the deltas from backpropagation
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].weights.size(); ++j) {
            for (size_t k = 0; k < layers[i].weights[j].size(); ++k) {
                layers[i].weights[j][k] -= learning_rate * output_deltas[i][j] * input_activations[i][k];
            }
        }
    }
}


#include "neural_network.h"

double learning_rate_decay(double initial_rate, int epoch, int max_epochs) {
    return initial_rate * (1 - (epoch / (double)max_epochs));
}

void NeuralNetwork::adjust_learning_rate(int epoch, int max_epochs) {
    learning_rate = learning_rate_decay(learning_rate, epoch, max_epochs);
}

// neural_network.cpp
#include "dropout.h"

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> input_data = input;
    Dropout dropout(0.5);  // Example dropout rate of 50%
    for (size_t i = 0; i < layers.size(); ++i) {
        std::vector<double> output(layers[i].neurons.size(), 0.0);
        for (size_t j = 0; j < layers[i].neurons.size(); ++j) {
            for (size_t k = 0; k < input_data.size(); ++k) {
                output[j] += input_data[k] * layers[i].weights[j][k];
            }
            output[j] += layers[i].biases[j];
            output = dropout.apply_dropout(output);  // Apply dropout
            layers[i].neurons[j] = activation_function(output[j], activation_functions[i]);
        }
        input_data = output;
    }
    return input_data;
}
#include "neural_network.h"
#include <cmath>
#include <iostream>

NeuralNetwork::NeuralNetwork(std::vector<int> sizes) {
    numLayers = sizes.size();
    layerSizes = sizes;
    
    weights.resize(numLayers - 1);
    biases.resize(numLayers - 1);
    activations.resize(numLayers);
    z_values.resize(numLayers - 1);

    for (int i = 0; i < numLayers - 1; ++i) {
        weights[i].resize(layerSizes[i + 1]);
        biases[i].resize(layerSizes[i + 1]);
        z_values[i].resize(layerSizes[i + 1]);
        activations[i].resize(layerSizes[i]);

        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            biases[i][j] = ((double) rand() / (RAND_MAX)) * 0.01;
            for (int k = 0; k < layerSizes[i]; ++k) {
                weights[i][j].push_back(((double) rand() / (RAND_MAX)) * 0.01);
            }
        }
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double NeuralNetwork::getActivation(int layerIndex, int neuronIndex) const {
    return activations[layerIndex][neuronIndex];
}
