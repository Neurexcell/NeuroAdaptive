// activation_functions.cpp
#include "activation_functions.h"
#include <cmath>

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double tanh(double x) {
    return std::tanh(x);
}

double activation_function(double x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::ReLU:
            return relu(x);
        case ActivationFunction::Sigmoid:
            return sigmoid(x);
        case ActivationFunction::Tanh:
            return tanh(x);
        default:
            return 0;
    }
}

