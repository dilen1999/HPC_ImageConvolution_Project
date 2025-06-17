<<<<<<< HEAD
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h> // OpenMP

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef vector<vector<float>> Matrix;

// Clamp helper
uchar clamp(float val)
{
    return min(255.0f, max(0.0f, val));
}

// Apply convolution with OpenMP
Mat applyConvolution(const Mat &input, const Matrix &kernel)
{
    int kSize = kernel.size();
    int offset = kSize / 2;
    Mat output = input.clone();

// Parallel outer loop (rows)
#pragma omp parallel for
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < kSize; ++ki)
            {
                for (int kj = 0; kj < kSize; ++kj)
                {
                    int ni = i + ki - offset;
                    int nj = j + kj - offset;
                    if (ni >= 0 && ni < input.rows && nj >= 0 && nj < input.cols)
                    {
                        sum += kernel[ki][kj] * input.at<uchar>(ni, nj);
                    }
                }
            }
            output.at<uchar>(i, j) = clamp(sum);
        }
    }

    return output;
}

int main()
{
    string input_file = "../Images/input.png";
    string output_file = "../Results/output_openmp.png";

    // Load image in grayscale
    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // Define kernel (e.g., edge detection)
    Matrix kernel = {
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}};

    auto start = high_resolution_clock::now();
    Mat result = applyConvolution(img, kernel);
    auto end = high_resolution_clock::now();

    imwrite(output_file, result);

    duration<double> diff = end - start;
    cout << "OpenMP Convolution completed in " << diff.count() << " seconds." << endl;

    return 0;
}
=======
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h> // OpenMP

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef vector<vector<float>> Matrix;

// Clamp helper
uchar clamp(float val) {
    return min(255.0f, max(0.0f, val));
}

// Apply convolution with OpenMP
Mat applyConvolution(const Mat& input, const Matrix& kernel) {
    int kSize = kernel.size();
    int offset = kSize / 2;
    Mat output = input.clone();

    // Parallel outer loop (rows)
    #pragma omp parallel for
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kSize; ++ki) {
                for (int kj = 0; kj < kSize; ++kj) {
                    int ni = i + ki - offset;
                    int nj = j + kj - offset;
                    if (ni >= 0 && ni < input.rows && nj >= 0 && nj < input.cols) {
                        sum += kernel[ki][kj] * input.at<uchar>(ni, nj);
                    }
                }
            }
            output.at<uchar>(i, j) = clamp(sum);
        }
    }

    return output;
}

int main() {
    string input_file = "../Images/input.png";
    string output_file = "../Results/output_openmp.png";

    // Load image in grayscale
    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // Define kernel (e.g., edge detection)
    Matrix kernel = {
        {0, -1, 0},
        {-1,  5, -1},
        {0, -1, 0}
    };

    auto start = high_resolution_clock::now();
    Mat result = applyConvolution(img, kernel);
    auto end = high_resolution_clock::now();

    imwrite(output_file, result);

    duration<double> diff = end - start;
    cout << "OpenMP Convolution completed in " << diff.count() << " seconds." << endl;

    return 0;
}
>>>>>>> 0af2a88 (feat: add openmp file and results)
