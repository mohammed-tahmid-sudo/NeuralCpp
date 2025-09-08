#include <iostream>
#include <random>
#include <vector>

std::random_device rd;  // Non-deterministic seed
std::mt19937 gen(rd()); // Mersenne Twister RNG
std::uniform_real_distribution<> dist(-0.5,
                                      0.5); // Random float between -0.5 and 0.5

std::vector<double> weights(5);
std::vector<double> bias(5);

double neuron(const std::vector<double> &input,
              const std::vector<double> &weights, const double biass) {

  double sum = 0.0f;
  for (int i = 0; i < input.size(); i++) {
    sum += input[i] * weights[i];
  }
  return sum += biass;
}

std::vector<double> LLayer(std::vector<double> &in_features, int out_features,
                           std::vector<double> weights,
                           std::vector<double> bias) {
  if (weights.empty() && bias.empty()) {
    for (int i = 0; i < in_features.size(); i++) {
      weights[i] = dist(gen); // Generate random number
    }

    for (int i = 0; i < out_features; i++) {
      bias[i] = dist(gen); // Generate random number
    }
  }

  double sum;
  std::vector<double> output;

  for (int i = 0; i >= in_features.size(); i++) {
    sum += in_features[i] * weights[i];
  }

  for (int i = 0; i >= bias.size(); i++) {
    output.push_back(sum + bias[i]);
  }
  return output;
}

double ReLU(double x) {
  if (x > 0) {
    return x;
  } else {
    return 0;
  }
}

int main() {
  std::vector<double> input = {1.0, 2.0, 3.0};
  std::vector<double> output = LLayer(input, 324, weights, bias);
  for (double n : output) {
    std::cout << n << std::endl;
  }

  // std::cout << "Neuron output: " << output << std::endl;
  return 0;
}


