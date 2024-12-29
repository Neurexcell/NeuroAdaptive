#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork {
private:
    int numLayers;
    std::vector<int> layerSizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> z_values;
    
public:
    NeuralNetwork(std::vector<int> sizes);
    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& expected_output);
    void updateWeights(double learningRate);
    
    double sigmoid(double x);
    double sigmoidDerivative(double x);

    double getActivation(int layerIndex, int neuronIndex) const;
};

#endif
