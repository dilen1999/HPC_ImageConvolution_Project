#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// Clamp helper for device
__device__ uchar clamp_cuda(float val) {
    return (val < 0.0f) ? 0 : ((val > 255.0f) ? 255 : (uchar)val);
}

// CUDA kernel
__global__ void cudaConvolution(uchar* input, uchar* output, float* kernel, int rows, int cols, int kSize) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = kSize / 2;

    if (i < rows && j < cols) {
        float sum = 0.0f;
        for (int ki = 0; ki < kSize; ++ki) {
            for (int kj = 0; kj < kSize; ++kj) {
                int ni = i + ki - offset;
                int nj = j + kj - offset;

                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                    sum += kernel[ki * kSize + kj] * input[ni * cols + nj];
                }
            }
        }
        output[i * cols + j] = clamp_cuda(sum);
    }
}

void runFilter(const Mat& img, const float* h_kernel, const string& filterName, int rows, int cols) {
    size_t imgSize = rows * cols * sizeof(uchar);
    size_t kernelSize = 3 * 3 * sizeof(float);

    uchar *d_input, *d_output;
    float *d_kernel;

    cudaMalloc((void**)&d_input, imgSize);
    cudaMalloc((void**)&d_output, imgSize);
    cudaMalloc((void**)&d_kernel, kernelSize);

    cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_kernel, rows, cols, 3);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    Mat result(rows, cols, CV_8UC1);
    cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    string output_file = "../Results/output_cuda_" + filterName + ".png";
    imwrite(output_file, result);
    cout << " CUDA " << filterName << " filter done: " << milliseconds / 1000.0 << " sec. Saved to " << output_file << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main() {
    string input_file = "../Images/input.png";
    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << " Error loading image!" << endl;
        return -1;
    }

    int rows = img.rows, cols = img.cols;
    cout << " Input image size: " << rows << " x " << cols << endl;

    // 1️ Sharpen
    float sharpenKernel[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
    runFilter(img, sharpenKernel, "sharpen", rows, cols);

    // 2️ Blur
    float blurKernel[9] = {
        1.0/9, 1.0/9, 1.0/9,
        1.0/9, 1.0/9, 1.0/9,
        1.0/9, 1.0/9, 1.0/9
    };
    runFilter(img, blurKernel, "blur", rows, cols);

    // 3️ Edge
    float edgeKernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    runFilter(img, edgeKernel, "edge", rows, cols);

    cout << " All CUDA filters done!" << endl;

    return 0;
}
