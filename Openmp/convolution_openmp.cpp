// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <chrono>
// #include <vector>
// #include <omp.h> // OpenMP

// using namespace std;
// using namespace cv;
// using namespace std::chrono;

// typedef vector<vector<float>> Matrix;

// // Clamp helper
// uchar clamp(float val) {
//     return min(255.0f, max(0.0f, val));
// }

// // Apply convolution with OpenMP
// Mat applyConvolution(const Mat& input, const Matrix& kernel) {
//     int kSize = kernel.size();
//     int offset = kSize / 2;
//     Mat output = input.clone();

//     // Parallel outer loop (rows)
//     #pragma omp parallel for
//     for (int i = 0; i < input.rows; ++i) {
//         for (int j = 0; j < input.cols; ++j) {
//             float sum = 0.0f;
//             for (int ki = 0; ki < kSize; ++ki) {
//                 for (int kj = 0; kj < kSize; ++kj) {
//                     int ni = i + ki - offset;
//                     int nj = j + kj - offset;
//                     if (ni >= 0 && ni < input.rows && nj >= 0 && nj < input.cols) {
//                         sum += kernel[ki][kj] * input.at<uchar>(ni, nj);
//                     }
//                 }
//             }
//             output.at<uchar>(i, j) = clamp(sum);
//         }
//     }

//     return output;
// }

// int main() {
//     string input_file = "../Images/input.png";
//     string output_file = "../Results/output_openmp.png";

//     // Load image in grayscale
//     Mat img = imread(input_file, IMREAD_GRAYSCALE);
//     if (img.empty()) {
//         cerr << "Error loading image!" << endl;
//         return -1;
//     }

//     // Define kernel (e.g., edge detection)
//     Matrix kernel = {
//         {0, -1, 0},
//         {-1,  5, -1},
//         {0, -1, 0}
//     };

//     auto start = high_resolution_clock::now();
//     Mat result = applyConvolution(img, kernel);
//     auto end = high_resolution_clock::now();

//     imwrite(output_file, result);

//     duration<double> diff = end - start;
//     cout << "OpenMP Convolution completed in " << diff.count() << " seconds." << endl;

//     return 0;
// }
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h> // OpenMP for parallel processing and omp_get_wtime()

using namespace std;
using namespace cv;

// Type alias for 2D kernel
typedef vector<vector<float>> Matrix;

// Clamp to ensure pixel values stay within 0–255
uchar clamp(float val) {
    return min(255.0f, max(0.0f, val));
}

// Convolution with OpenMP parallelism
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
                        const uchar* inputRow = input.ptr<uchar>(ni);
                        sum += kernel[ki][kj] * inputRow[nj];
                    }
                }
            }

            uchar* outputRow = output.ptr<uchar>(i);
            outputRow[j] = clamp(sum);
        }
    }

    return output;
}

int main() {
    string input_file = "../Images/input.png";
    string output_file = "../Results/output_openmp.png";

    // Load grayscale image
    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // Define a sharpening kernel
    Matrix kernel = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    // Time the convolution using omp_get_wtime
    double start = omp_get_wtime();
    Mat result = applyConvolution(img, kernel);
    double end = omp_get_wtime();

    // Save result
    imwrite(output_file, result);

    cout << "OpenMP Convolution completed in " << (end - start) << " seconds using "
         << omp_get_max_threads() << " threads." << endl;

    return 0;
}