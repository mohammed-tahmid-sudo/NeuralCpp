#include <iostream>
#include <random>
#include <vector>

class Neuron {
public:
  float neuron(const std::vector<double>& input) {
    float sum = 0.0f;

    std::vector<double> output;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(-0.01, 0.01);

    for (int i = 0; i < input.size(); i++) {
      double weight = uniform_dist(gen);
      double bias = uniform_dist(gen);
       sum += input[i] * weight + bias;

    }
    return sum;
  }
};

int main() {
    Neuron n;
    std::vector<double> input = {1.0, 2.0, 3.0};
    float output = n.neuron(input);                    //////////////////// changed output type from vector<double> to float

    std::cout << "Neuron output: " << output << std::endl;
    return 0;
}
