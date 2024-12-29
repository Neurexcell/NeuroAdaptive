#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& architecture);
    std::vector<double> forward(const std::vector<double>& input);
    void train(const std::vector<std::vector<double>>& data);

private:
    struct Layer {
        std::vector<std::vector<double>> neurons;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
    };

    std::vector<Layer> layers;

    static double relu(double x);
    static double sigmoid(double x);
    void initialize_weights();
};

#endif  // NEURAL_NETWORK_HPP

// neural_network.hpp

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& architecture);
    
    // Perform a forward pass with the given input
    std::vector<double> forward(const std::vector<double>& input);
    
    // Training function
    void train(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, int epochs, double learning_rate);
    
private:
    struct Layer {
        std::vector<std::vector<double>> neurons;  // Neurons matrix
        std::vector<std::vector<double>> weights;  // Weights matrix
        std::vector<double> biases;  // Bias for each neuron
    };

    std::vector<Layer> layers;

    static double relu(double x);  // ReLU activation function
    static double sigmoid(double x); // Sigmoid activation (for later use)
    static double relu_derivative(double x); // Derivative of ReLU for backpropagation
    static double sigmoid_derivative(double x); // Derivative of Sigmoid for backpropagation

    void initialize_weights();  // Initialize weights with small random values
    double calculate_error(const std::vector<double>& predicted, const std::vector<double>& actual);  // Calculate error
    void update_weights(const std::vector<std::vector<double>>& output_deltas, const std::vector<std::vector<double>>& input_activations, double learning_rate);  // Update weights and biases
};

#endif  // NEURAL_NETWORK_HPP

