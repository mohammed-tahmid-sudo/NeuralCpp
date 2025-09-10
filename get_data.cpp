#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class test_data {
public:
    std::vector<std::vector<double>> data() {
        std::ifstream file("/home/tahmid/NeuralCpp/data/mnist_test.csv");
        if (!file.is_open()) {
            std::cerr << "Could not open file\n";
            return {}; // return empty vector
        }

        std::string line;
        std::vector<std::vector<double>> result;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;

            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell)); // convert string to double
                } catch (const std::invalid_argument &e) {
                    row.push_back(0.0); // fallback if conversion fails
                }
            }

            result.push_back(row);
        }

        file.close();
        return result;
    }
};

int main() {
    test_data td;
    std::vector<std::vector<double>> mydata = td.data();

    std::cout << "Loaded " << mydata.size() << " rows.\n";
    if (!mydata.empty())
        std::cout << "First row size: " << mydata[0].size() << "\n";

    return 0;
}

