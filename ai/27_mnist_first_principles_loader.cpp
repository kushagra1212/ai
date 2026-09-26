#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>

// Helper to convert big-endian 32-bit integer to native little-endian
uint32_t read_big_endian_uint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

struct MNISTSample {
    uint8_t label;
    std::vector<std::vector<double>> pixels; // 28x28 normalized to [0.0, 1.0]
};

// Pure C++ First-Principles Binary Loader (Zero External Libraries!)
std::vector<MNISTSample> load_mnist_raw(const std::string& images_path, 
                                       const std::string& labels_path, 
                                       int max_samples = 10) 
{
    std::ifstream img_file(images_path, std::ios::binary);
    std::ifstream lbl_file(labels_path, std::ios::binary);

    if (!img_file.is_open() || !lbl_file.is_open()) {
        std::cerr << "Error: Could not open MNIST files at " << images_path << "\n";
        return {};
    }

    uint32_t img_magic = read_big_endian_uint32(img_file);
    uint32_t num_images = read_big_endian_uint32(img_file);
    uint32_t num_rows = read_big_endian_uint32(img_file);
    uint32_t num_cols = read_big_endian_uint32(img_file);

    uint32_t lbl_magic = read_big_endian_uint32(lbl_file);
    uint32_t num_labels = read_big_endian_uint32(lbl_file);

    std::cout << "MNIST Header Info:\n";
    std::cout << "  Images Magic : " << img_magic << " (Expected 2051)\n";
    std::cout << "  Total Images : " << num_images << "\n";
    std::cout << "  Dimensions   : " << num_rows << " x " << num_cols << " pixels\n";
    std::cout << "  Total Labels : " << num_labels << "\n\n";

    int load_count = std::min(static_cast<uint32_t>(max_samples), num_images);
    std::vector<MNISTSample> samples(load_count);

    for (int i = 0; i < load_count; ++i) {
        uint8_t lbl = 0;
        lbl_file.read(reinterpret_cast<char*>(&lbl), 1);
        samples[i].label = lbl;

        samples[i].pixels.assign(num_rows, std::vector<double>(num_cols, 0.0));
        for (uint32_t r = 0; r < num_rows; ++r) {
            for (uint32_t c = 0; c < num_cols; ++c) {
                uint8_t px = 0;
                img_file.read(reinterpret_cast<char*>(&px), 1);
                samples[i].pixels[r][c] = static_cast<double>(px) / 255.0; // normalize to [0.0, 1.0]
            }
        }
    }

    return samples;
}

void print_mnist_ascii(const MNISTSample& sample, int sample_idx) {
    std::cout << "Sample #" << sample_idx << " | True Label: [" << static_cast<int>(sample.label) << "]\n";
    std::cout << "--------------------------------------------------------\n";
    for (int r = 0; r < 28; ++r) {
        std::cout << "  ";
        for (int c = 0; c < 28; ++c) {
            double p = sample.pixels[r][c];
            if (p > 0.75) std::cout << "##";
            else if (p > 0.40) std::cout << "**";
            else if (p > 0.10) std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=========================================================================\n";
    std::cout << " PROGRAM 27: FIRST-PRINCIPLES MNIST RAW BINARY LOADER (C++17)\n";
    std::cout << "=========================================================================\n\n";

    std::string img_path = "data/mnist/train-images-idx3-ubyte";
    std::string lbl_path = "data/mnist/train-labels-idx1-ubyte";

    auto samples = load_mnist_raw(img_path, lbl_path, 3);

    for (size_t i = 0; i < samples.size(); ++i) {
        print_mnist_ascii(samples[i], i);
    }

    std::cout << "=========================================================================\n";
    std::cout << "SUCCESS: Real human handwritten images loaded directly from raw disk bytes!\n";
    std::cout << "=========================================================================\n";

    return 0;
}
