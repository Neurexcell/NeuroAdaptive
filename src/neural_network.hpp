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

