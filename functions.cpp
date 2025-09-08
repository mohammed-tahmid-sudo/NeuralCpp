#include <iostream>
#include <random>
#include <vector>

std::random_device rd;  // Non-deterministic seed
std::mt19937 gen(rd()); // Mersenne Twister RNG
std::uniform_real_distribution<> dist(-0.5,
                                      0.5); // Random float between -0.5 and 0.5

std::vector<double> weights_and_bias_generator(int length) {
  std::vector<double> numbers;
  for (int i = 0; i < length; i++)
    numbers.push_back(dist(gen));
  return numbers;
}

std::vector<double> weights = weights_and_bias_generator(5);
std::vector<double> bias = weights_and_bias_generator(5);

std::vector<double> LLayer(std::vector<double> &in_features, int out_features,
                           std::vector<double> weights,
                           std::vector<double> bias) {
  std::vector<double> output;

  for (int i = 0; i < out_features; i++) {
    double sum = 0.0;
    for (int j = 0; j < in_features.size(); j++) {
      sum += in_features[j] * weights[j % weights.size()];
    }
    output.push_back(sum + bias[i % bias.size()]);
  }
  return output;
}

double ReLU(double x) { return x > 0 ? x : 0; }

double cross_entropy_loss(const std::vector<double> &y_true,
                          const std::vector<double> &y_pred) {
  double loss = 0.0;
  for (size_t i = 0; i < y_true.size(); i++) {
    loss += y_true[i] * std::log(y_pred[i]);
  }
  return -loss; // cross-entropy is negative sum
}
void sgd_update(std::vector<double> &weights, const std::vector<double> &grads,
                double lr) {
  for (size_t i = 0; i < weights.size(); i++) {
    weights[i] -= lr * grads[i];
  }
}

int main() {
  std::vector<double> input = {1.0, 2.0, 3.0};
  std::vector<double> output = LLayer(input, 3, weights, bias);

  for (double n : output) {
    std::cout << ReLU(n) << std::endl;
  }

  return 0;
}
