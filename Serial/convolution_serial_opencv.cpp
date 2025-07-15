#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef vector<vector<float>> Matrix;

// Clamp helper
uchar clamp(float val)
{
    return min(255.0f, max(0.0f, val));
}

// Apply convolution
Mat applyConvolution(const Mat &input, const Matrix &kernel)
{
    int kSize = kernel.size();
    int offset = kSize / 2;
    Mat output = input.clone();

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
    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << " Error loading image!" << endl;
        return -1;
    }

    cout << " Input image loaded: " << img.rows << " x " << img.cols << endl;

    // ------------------------------
    // 1️ Sharpen kernel
    Matrix sharpenKernel = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}};

    auto start1 = high_resolution_clock::now();
    Mat sharpenResult = applyConvolution(img, sharpenKernel);
    auto end1 = high_resolution_clock::now();
    imwrite("../Results/output_serial_sharpen.png", sharpenResult);
    duration<double> diff1 = end1 - start1;
    cout << " Sharpen filter completed in " << diff1.count() << " seconds." << endl;

    // ------------------------------
    // 2️ Blur kernel (simple box blur)
    Matrix blurKernel = {
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9}};

    auto start2 = high_resolution_clock::now();
    Mat blurResult = applyConvolution(img, blurKernel);
    auto end2 = high_resolution_clock::now();
    imwrite("../Results/output_serial_blur.png", blurResult);
    duration<double> diff2 = end2 - start2;
    cout << " Blur filter completed in " << diff2.count() << " seconds." << endl;

    // ------------------------------
    // 3️ Edge detection kernel
    Matrix edgeKernel = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}};

    auto start3 = high_resolution_clock::now();
    Mat edgeResult = applyConvolution(img, edgeKernel);
    auto end3 = high_resolution_clock::now();
    imwrite("../Results/output_serial_edge.png", edgeResult);
    duration<double> diff3 = end3 - start3;
    cout << " Edge detection filter completed in " << diff3.count() << " seconds." << endl;

    cout << " All filters done. Check your Results folder!" << endl;

    return 0;
}
