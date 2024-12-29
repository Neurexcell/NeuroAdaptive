#ifndef DATA_HANDLER_HPP
#define DATA_HANDLER_HPP

#include <vector>

class DataHandler {
public:
    static std::vector<std::vector<double>> load_data(const std::string& filepath);
    static std::vector<double> normalize(const std::vector<double>& data);
};

#endif  // DATA_HANDLER_HPP

