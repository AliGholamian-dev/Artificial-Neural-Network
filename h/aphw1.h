#include <iostream>
#include <vector>

std::vector<std::vector<double>> getData(const char *filename, bool add_bias = false);
void displayDataset(std::vector<std::vector<double>> data, bool has_bias = false);
double h(std::vector<double> features, std::vector<double> w);
double J(std::vector<std::vector<double>> data, std::vector<size_t> indices, std::vector<double> w);
std::vector<double> fitOneEpoch(std::vector<std::vector<double>> data, std::vector<double> w0, double lr = 0.01, size_t batch_size = 8);
std::vector<double> fit(std::vector<std::vector<double>> data, std::vector<double> w0, double lr = 0.01, size_t epochs = 10, size_t batch_size = 8, bool verbose = false);
std::vector<double> predict(std::vector<std::vector<double>> data, std::vector<double> w, bool verbose = false);