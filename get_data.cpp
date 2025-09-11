#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

std::pair<std::vector<std::vector<double>>, std::vector<int>>
load_mnist_csv_debug(const std::string &path, bool normalize = true, std::size_t expected_pixels = 784) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Could not open: " + path);

    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    std::string line;
    size_t lineno = 0;

    while (std::getline(file, line)) {
        ++lineno;
        if (line.empty()) continue;

        // show raw lines for the first 3
        if (lineno <= 3) {
            std::cout << "RAW LINE " << lineno << ": [" << line << "]\n";
        }

        std::stringstream ss(line);
        std::string cell;

        // read label
        if (!std::getline(ss, cell, ',')) {
            std::cerr << "Parse fail at line " << lineno << "\n";
            continue;
        }
        // trim cell
        cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(), [](unsigned char ch){ return !std::isspace(ch); }));
        cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), cell.end());

        int lab = 0;
        try { lab = std::stoi(cell); } catch (...) { lab = 0; }
        labels.push_back(lab);

        // pixels
        std::vector<double> pix;
        pix.reserve(expected_pixels);
        while (std::getline(ss, cell, ',')) {
            // trim
            cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(), [](unsigned char ch){ return !std::isspace(ch); }));
            cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), cell.end());
            try {
                double v = std::stod(cell);
                if (normalize) v /= 255.0;
                pix.push_back(v);
            } catch (...) {
                pix.push_back(0.0);
            }
        }

        if (pix.size() != expected_pixels) {
            std::cerr << "Warning: row " << lineno << " has " << pix.size()
                      << " pixels (expected " << expected_pixels << ")\n";
            // keep going — useful for debugging
        }

        images.push_back(std::move(pix)); // move ensures separate storage
    }

    // debug prints
    std::cout << "\nParsed first 3 samples (label + first 10 pixels):\n";
    for (size_t i = 0; i < images.size() && i < 3; ++i) {
        std::cout << "Label[" << i << "] = " << labels[i] << "  Pixels: ";
        for (size_t j = 0; j < images[i].size() && j < 10; ++j) std::cout << images[i][j] << ' ';
        std::cout << '\n';
    }

    // check if all images are identical to the first
    bool all_same = true;
    if (!images.empty()) {
        for (size_t i = 1; i < images.size(); ++i) {
            if (images[i] != images[0]) { all_same = false; break; }
        }
    }
    std::cout << (all_same ? "All images ARE identical\n" : "Images differ (not identical)\n");

    return {std::move(images), std::move(labels)};
}

int main() {
    auto [images, labels] = load_mnist_csv_debug("/home/tahmid/NeuralCpp/data/mnist_test.csv", true);
    // std::cout << "Loaded: " << images.size() << " samples\n";
    for (auto n : labels) {
        std::cout << n << std::endl; 
    }
    return 0;
}

