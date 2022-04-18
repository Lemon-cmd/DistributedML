#ifndef __MNIST_PARSER__
#define __MNIST_PARSER__

#include <iostream>
#include <fstream>

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

float *char_to_int(char *val, int size)
{
    float *out = new float[size];
    for (int i = 0; i < size; i++)
    {
        out[i] = val[i];
    }

    return out;
}

// std::vector<float *> &img, std::vector<float *> &labels
void load_mnist(size_t batch_size, const char *image_filename, const char *label_filename)
{
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char *>(&magic), 4);
    magic = swap_endian(magic);

    if (magic != 2051)
    {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    label_file.read(reinterpret_cast<char *>(&magic), 4);
    magic = swap_endian(magic);

    if (magic != 2049)
    {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    image_file.read(reinterpret_cast<char *>(&num_items), 4);
    num_items = swap_endian(num_items);

    label_file.read(reinterpret_cast<char *>(&num_labels), 4);
    num_labels = swap_endian(num_labels);

    if (num_items != num_labels)
    {
        std::cout << "image file nums should equal to label num" << std::endl;
        return;
    }

    // labels.resize(num_items);
    // img.resize(num_items);

    image_file.read(reinterpret_cast<char *>(&rows), 4);
    rows = swap_endian(rows);

    image_file.read(reinterpret_cast<char *>(&cols), 4);
    cols = swap_endian(cols);

    std::cout << "image and label num is: " << num_items << std::endl;
    std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

    char label;
    char *pixels = new char[rows * cols];
    float *img = new float[rows * cols];

    size_t partitions = num_items / batch_size;

    for (int item_id = 0; item_id < num_items; ++item_id)
    {
        // read image pixel
        image_file.read(pixels, rows * cols);

        // read label
        label_file.read(&label, 1);

        std::string sLabel = std::to_string(int(label));

        std::cout << "label is: " << sLabel << std::endl;
        // std::cout << "shape: " << rows << ' ' << cols << std::endl;

        // img[item_id] = Eigen::TensorMap<Eigen::Tensor<float, 3>>(char_to_int(pixels, rows * cols), Eigen::DSizes<ptrdiff_t, 3>{1, rows, cols});
        // labels[item_id] = Eigen::Tensor<float, 3>{1, 1, 10};
        // labels[item_id](0, 0, label) = 1.0;
    }

    delete[] pixels;
}

int main()
{
    char *buff1 = "../data/mnist-train-images";
    char *buff2 = "../data/mnist-train-labels";
    load_mnist(buff1, buff2);
}

#endif
