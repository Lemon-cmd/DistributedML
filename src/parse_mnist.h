#ifndef __MNIST_PARSER__
#define __MNIST_PARSER__

#include "cu_mat.h"

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

void load_mnist(const char *image_filename,
                const char *label_filename,
                size_t batch_size,
                std::vector<Matrix> &images,
                std::vector<Matrix> &labels)
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

    image_file.read(reinterpret_cast<char *>(&rows), 4);
    rows = swap_endian(rows);

    image_file.read(reinterpret_cast<char *>(&cols), 4);
    cols = swap_endian(cols);

    std::cout << "image and label num is: " << num_items << std::endl;
    std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

    char label;
    char *pixels = new char[rows * cols];

    std::vector<std::vector<float>> fpixels(num_items,
                                            std::vector<float>(rows * cols, 0.0));

    std::vector<float> flabels(num_items, 0.0);

    for (int item_id = 0; item_id < num_items; ++item_id)
    {
        // read image pixel
        image_file.read(pixels, rows * cols);

        for (uint j = 0; j < rows * cols; j++)
        {
            fpixels[item_id][j] = pixels[j];
        }

        // read label
        label_file.read(&label, 1);

        // if image is of a 7 then true
        if (label == 7)
            flabels[item_id] = 1.0;
        else
            flabels[item_id] = 0.0;

        // std::string sLabel = std::to_string(int(label));
        // std::cout << "label is: " << sLabel << std::endl;
    }

    delete[] pixels;

    /* Partition Data */

    for (uint j = 0; j < num_items; j += batch_size)
    {
        std::vector<float> lab_slice(flabels.begin() + j, flabels.begin() + j + batch_size);

        std::vector<float> slice_data;
        std::vector<std::vector<float>> slice(fpixels.begin() + j, fpixels.begin() + j + batch_size);

        for (uint i = 0; i < batch_size; i++)
        {
            for (uint k = 0; k < rows * cols; k++)
            {
                slice_data.push_back(slice[i][k]);
            }
        }

        Matrix Y(batch_size, 1, lab_slice);
        Matrix X(batch_size, rows * cols, slice_data);

        images.push_back(X);
        labels.push_back(Y);
    }

    std::cout << "Partitions: " << num_items / batch_size << std::endl;
}

#endif
