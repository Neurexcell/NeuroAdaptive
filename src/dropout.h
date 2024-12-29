// dropout.h
#ifndef DROPOUT_H
#define DROPOUT_H

#include <vector>

class Dropout {
public:
    Dropout(double dropout_rate);
    std::vector<double> apply_dropout(const std::vector<double>& input);
private:
    double dropout_rate;
};

#endif // DROPOUT_H

