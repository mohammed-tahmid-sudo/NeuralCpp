#include "get_data.cpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// Small fully-connected NN with ReLU and softmax + SGD optimizer.
// Assumes load_mnist_csv_debug(path, normalize) returns
// std::pair<std::vector<std::vector<double>>, std::vector<int>>
// where each image is already a flattened vector of 784 doubles and labels are 0..9.

static std::random_device rd_global;
static std::mt19937 gen_global(rd_global());
static std::uniform_real_distribution<> init_dist(-0.5, 0.5);

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

// Utility: create zero-filled vector/matrix
static Vec zeros_vec(size_t n) { return Vec(n, 0.0); }
static Mat zeros_mat(size_t r, size_t c) { return Mat(r, Vec(c, 0.0)); }

// Dense layer (stores weights, bias) and supports forward/backward
struct Dense {
  Mat W; // out x in
  Vec b; // out
  Vec input_cache; // flattened input stored for backward
  Mat dW; // gradient accumulator for weights
  Vec db;

  Dense() = default;
  Dense(size_t in_features, size_t out_features) {
    W = Mat(out_features, Vec(in_features));
    for (size_t i = 0; i < out_features; ++i)
      for (size_t j = 0; j < in_features; ++j)
        W[i][j] = init_dist(gen_global);
    b = Vec(out_features);
    for (size_t i = 0; i < out_features; ++i) b[i] = init_dist(gen_global);

    dW = zeros_mat(out_features, in_features);
    db = zeros_vec(out_features);
  }

  Vec forward(const Vec &x) {
    input_cache = x;
    Vec out(W.size());
    for (size_t i = 0; i < W.size(); ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < x.size(); ++j) sum += x[j] * W[i][j];
      out[i] = sum + b[i];
    }
    return out;
  }

  // grad_output has size = out_features
  Vec backward(const Vec &grad_output) {
    size_t out_f = W.size();
    size_t in_f = W[0].size();
    // reset grads
    for (size_t i = 0; i < out_f; ++i) {
      db[i] += grad_output[i];
      for (size_t j = 0; j < in_f; ++j) dW[i][j] += grad_output[i] * input_cache[j];
    }
    // compute grad wrt input to pass further back
    Vec grad_input(in_f, 0.0);
    for (size_t j = 0; j < in_f; ++j) {
      double s = 0.0;
      for (size_t i = 0; i < out_f; ++i) s += W[i][j] * grad_output[i];
      grad_input[j] = s;
    }
    return grad_input;
  }

  // apply SGD update with learning rate lr and batch size for averaging
  void apply_grad(double lr, double batch_size) {
    double inv_bs = 1.0 / batch_size;
    for (size_t i = 0; i < W.size(); ++i) {
      for (size_t j = 0; j < W[0].size(); ++j) {
        W[i][j] -= lr * (dW[i][j] * inv_bs);
        dW[i][j] = 0.0; // reset after update
      }
      b[i] -= lr * (db[i] * inv_bs);
      db[i] = 0.0;
    }
  }
};

// ReLU activation (elementwise)
struct ReLU {
  Vec mask;
  Vec forward(const Vec &x) {
    mask.resize(x.size());
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      mask[i] = x[i] > 0 ? 1.0 : 0.0;
      out[i] = x[i] > 0 ? x[i] : 0.0;
    }
    return out;
  }
  Vec backward(const Vec &grad_output) {
    Vec out(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) out[i] = grad_output[i] * mask[i];
    return out;
  }
};

// Softmax + cross-entropy loss combined. Returns loss and gradient wrt logits.
static std::pair<double, Vec> softmax_cross_entropy_with_logits(const Vec &logits, int true_label) {
  double maxv = *std::max_element(logits.begin(), logits.end());
  Vec exps(logits.size());
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    exps[i] = std::exp(logits[i] - maxv);
    sum += exps[i];
  }
  Vec probs(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) probs[i] = exps[i] / sum;
  double loss = -std::log(std::max(probs[true_label], 1e-12));
  // gradient: probs - one_hot
  Vec grad(logits.size());
  for (size_t i = 0; i < probs.size(); ++i) grad[i] = probs[i];
  grad[true_label] -= 1.0;
  return {loss, grad};
}

int argmax(const Vec &v) {
  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

// Simple model container: Dense(784->128) - ReLU - Dense(128->128) - ReLU - Dense(128->10)
struct SimpleNN {
  Dense d1, d2, d3;
  ReLU r1, r2;
  SimpleNN() : d1(784, 128), d2(128, 128), d3(128, 10) {}

  Vec forward(const Vec &x) {
    Vec z1 = d1.forward(x);
    Vec a1 = r1.forward(z1);
    Vec z2 = d2.forward(a1);
    Vec a2 = r2.forward(z2);
    Vec z3 = d3.forward(a2);
    return z3; // logits
  }

  // Backprop given gradient of loss wrt logits
  void backward(const Vec &grad_logits) {
    Vec g3 = d3.backward(grad_logits);
    Vec g2 = r2.backward(g3);
    Vec g2_in = d2.backward(g2);
    Vec g1 = r1.backward(g2_in);
    (void) d1.backward(g1); // we don't use returned grad further
  }

  void apply_grad(double lr, double batch_size) {
    d1.apply_grad(lr, batch_size);
    d2.apply_grad(lr, batch_size);
    d3.apply_grad(lr, batch_size);
  }
};

int main() {
  // Load MNIST (expects flattened images). Adjust the loader if your get_data.cpp returns
  // images as 2D arrays — convert to flattened form before using.
  auto train_pair = load_mnist_csv_debug("/home/tahmid/NeuralCpp/data/mnist_train.csv", true);
  auto test_pair = load_mnist_csv_debug("/home/tahmid/NeuralCpp/data/mnist_test.csv", true);

  // The loader is expected to return: pair<vector<vector<double>>, vector<int>>
  auto &train_x = train_pair.first; // [N][784]
  auto &train_y = train_pair.second; // [N]
  auto &test_x = test_pair.first;
  auto &test_y = test_pair.second;

  if (train_x.empty()) {
    std::cerr << "Training data empty. Check your loader.\n";
    return 1;
  }

  // Quick sanity check: image size
  size_t img_size = train_x[0].size();
  if (img_size != 784) std::cerr << "Warning: expected 784 pixels, got " << img_size << '\n';

  SimpleNN net;

  // Training hyperparams (small so it runs reasonably fast)
  const int epochs = 3;
  const size_t batch_size = 32;
  const double lr = 0.01;

  size_t train_N = train_x.size();

  for (int epoch = 1; epoch <= epochs; ++epoch) {
    double epoch_loss = 0.0;
    size_t correct = 0;

    // shuffle indices
    std::vector<size_t> idx(train_N);
    for (size_t i = 0; i < train_N; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), gen_global);

    for (size_t start = 0; start < train_N; start += batch_size) {
      size_t end = std::min(start + batch_size, train_N);
      size_t actual_bs = end - start;

      // For each sample in batch: forward, compute loss and accumulate gradients
      for (size_t b = start; b < end; ++b) {
        size_t i = idx[b];
        const Vec &x = train_x[i];
        int label = train_y[i];

        Vec logits = net.forward(x);
        auto [loss, dlogits] = softmax_cross_entropy_with_logits(logits, label);
        epoch_loss += loss;
        if (argmax(logits) == label) ++correct;

        net.backward(dlogits);
      }

      // After processing the batch, update parameters (gradients accumulated inside layers)
      net.apply_grad(lr, static_cast<double>(actual_bs));
    }

    double avg_loss = epoch_loss / static_cast<double>(train_N);
    double accuracy = static_cast<double>(correct) / static_cast<double>(train_N);
    std::cout << "Epoch " << epoch << ": loss=" << avg_loss << " accuracy=" << accuracy << "\n";
  }

  // Quick test evaluation (no grad)
  size_t test_N = test_x.size();
  size_t test_correct = 0;
  for (size_t i = 0; i < test_N; ++i) {
    Vec logits = net.forward(test_x[i]);
    if (argmax(logits) == test_y[i]) ++test_correct;
  }
  std::cout << "Test accuracy: " << (double)test_correct / test_N << "\n";

  return 0;
}

