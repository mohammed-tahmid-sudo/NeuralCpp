#include <iostream>
#include <random>
#include <vector>

std::random_device rd;  // Non-deterministic seed
std::mt19937 gen(rd()); // Mersenne Twister RNG
std::uniform_real_distribution<> dist(-0.5,
                                      0.5); // Random float between -0.5 and 0.5

std::vector<std::vector<double>> weights;
std::vector<double> bias;

std::vector<double> flatten(std::vector<std::vector<double>> input) {
  std::vector<double> result;
  for (const auto &row : input) {
    for (double val : row) {
      result.push_back(val);
    }
  }
  return result;
}

std::vector<double> Dense(std::vector<double> in_features, int out_features,
                          std::vector<std::vector<double>> weights = {},
                          std::vector<double> bias = {}) {
  int in_features_size = in_features.size();

  // Random generator
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> dist(-0.5, 0.5);

  // Initialize weights if empty
  if (weights.empty()) {
    weights.resize(out_features, std::vector<double>(in_features_size));
    for (int i = 0; i < out_features; i++) {
      for (int j = 0; j < in_features_size; j++) {
        weights[i][j] = dist(gen);
      }
    }
  }

  // Initialize bias if empty
  if (bias.empty()) {
    bias.resize(out_features);
    for (int i = 0; i < out_features; i++) {
      bias[i] = dist(gen);
    }
  }

  // Forward pass
  std::vector<double> output(out_features, 0.0);
  for (int i = 0; i < out_features; i++) {
    for (int j = 0; j < in_features_size; j++) {
      output[i] += in_features[j] * weights[i][j];
    }
    output[i] += bias[i];
  }

  return output;
}

double ReLU(double x) { return x > 0 ? x : 0; }

std::vector<double> relu_vec(const std::vector<double> &v) {
  std::vector<double> out(v.size());
  for (size_t i = 0; i < v.size(); i++)
    out[i] = ReLU(v[i]);
  return out;
}

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

std::vector<double> model(std::vector<std::vector<double>> weights,
                          std::vector<double> bias,
                          std::vector<std::vector<double>> input) {
  std::vector<double> temp;
  std::vector<double> output;

  static std::vector<std::vector<double>> W1, W2, W3;
  static std::vector<double> B1, B2, B3;

  auto h1 = relu_vec(Dense(flatten(input), 128, W1, B1));
  auto h2 = relu_vec(Dense(h1, 128, W2, B2));
  auto out = Dense(h2, 10, W3, B3); // no ReLU here if classification

  return out;
}

int main() {
  std::vector<std::vector<double>> input = {
      {213.12}, {1234}, {1234}, {1234}, {1234},
      {1234},   {1234}, {1234}, {1234}, {1234},
  };

  std::vector<double> model_output = model(weights, bias, input);

  for (auto i : model_output) {
    std::cout << i << std::endl;
  }

  return 0;
}
