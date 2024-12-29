// activation_functions.h
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

enum class ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh
};

double relu(double x);
double sigmoid(double x);
double tanh(double x);
double activation_function(double x, ActivationFunction func);

#endif // ACTIVATION_FUNCTIONS_H

