// serialization.h
#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "neural_network.h"

void save_model(const NeuralNetwork& network, const std::string& filename);
void load_model(NeuralNetwork& network, const std::string& filename);

#endif // SERIALIZATION_H

