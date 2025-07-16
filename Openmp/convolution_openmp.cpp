#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h> 

using namespace std;
using namespace cv;

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

    #pragma omp parallel for schedule(static)
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

    // Load image
    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << " Error loading image!" << endl;
        return -1;
    }

    cout << " Input image loaded: " << img.rows << " x " << img.cols << endl;
    cout << " OpenMP threads: " << omp_get_max_threads() << endl;

    // 1️ Sharpen kernel
    Matrix sharpenKernel = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };

    double start1 = omp_get_wtime();
    Mat sharpenResult = applyConvolution(img, sharpenKernel);
    double end1 = omp_get_wtime();
    imwrite("../Results/output_openmp_sharpen.png", sharpenResult);
    cout << " Sharpen filter: " << (end1 - start1) << " sec, Threads: "
         << omp_get_max_threads() << endl;

    // 2️ Blur kernel
    Matrix blurKernel = {
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9}
    };

    double start2 = omp_get_wtime();
    Mat blurResult = applyConvolution(img, blurKernel);
    double end2 = omp_get_wtime();
    imwrite("../Results/output_openmp_blur.png", blurResult);
    cout << " Blur filter: " << (end2 - start2) << " sec, Threads: "
         << omp_get_max_threads() << endl;

    // 3️ Edge detection kernel
    Matrix edgeKernel = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    double start3 = omp_get_wtime();
    Mat edgeResult = applyConvolution(img, edgeKernel);
    double end3 = omp_get_wtime();
    imwrite("../Results/output_openmp_edge.png", edgeResult);
    cout << " Edge filter: " << (end3 - start3) << " sec, Threads: "
         << omp_get_max_threads() << endl;

    cout << " All OpenMP filters done!" << endl;
    return 0;
}
