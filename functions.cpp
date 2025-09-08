#include <iostream>
#include <random>
#include <vector>




float neuron(const std::vector<double>& input, const std::vector<double>& weights, const double biass) {
             
    float sum = 0.0f;

    std::vector<double> output;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform_dist(-0.01, 0.01);

    for (int i = 0; i < input.size(); i++) {
      // double weight = uniform_dist(gen);
      // double bias = uniform_dist(gen);
       sum += input[i] * weights[i];

    }
    return sum += biass;
  }

float ReLU(float x) {
        if (x > 0) {
                return x;
        } else {
                return 0;
        }       
}

int main() {
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> weights = {3.0, -345.0, -33.0};
    double bias = -5.333333;
    float output = ReLU(neuron(input, weights, bias  ));            
    std::cout << "Neuron output: " << output << std::endl;
    return 0;
}
