// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include "activation_functions.h" // For activation function enum

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& architecture, 
                  const std::vector<ActivationFunction>& activation_functions);
    void adjust_learning_rate(int epoch, int max_epochs);
    double learning_rate;  // For dynamic learning rate
    // Add any other methods and attributes for your network
private:
    std::vector<ActivationFunction> activation_functions;
    // Layers, weights, and other properties of your network
};

#endif // NEURAL_NETWORK_H

