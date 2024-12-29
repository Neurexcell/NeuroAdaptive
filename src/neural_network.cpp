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
