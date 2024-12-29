// serialization.cpp
#include "serialization.h"
#include <fstream>
#include <nlohmann/json.hpp>  // Include JSON for serialization

using json = nlohmann::json;

void save_model(const NeuralNetwork& network, const std::string& filename) {
    std::ofstream file(filename);
    json j;
    // Save network parameters like weights, biases in JSON format
    // Example: j["weights"] = network.get_weights(); // Adjust based on your class
    file << j.dump();
}

void load_model(NeuralNetwork& network, const std::string& filename) {
    std::ifstream file(filename);
    json j;
    file >> j;
    // Load network parameters from JSON and set in network
    // Example: network.set_weights(j["weights"]);
}

