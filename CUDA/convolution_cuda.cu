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

int main() {
    string input_file = "../Images/input.png";
    string output_file = "../Results/output_cuda.png";

    Mat img = imread(input_file, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "❌ Error loading image!" << endl;
        return -1;
    }

    int rows = img.rows, cols = img.cols;
    size_t imgSize = rows * cols * sizeof(uchar);
    size_t kernelSize = 3 * 3 * sizeof(float);

    float h_kernel[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    // Allocate device memory
    uchar *d_input, *d_output;
    float *d_kernel;

    cudaMalloc((void**)&d_input, imgSize);
    cudaMalloc((void**)&d_output, imgSize);
    cudaMalloc((void**)&d_kernel, kernelSize);

    cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);

    // Define grid/block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    // Time execution
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

    // Copy back result
    Mat result(rows, cols, CV_8UC1);
    cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    imwrite(output_file, result);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    cout << "✅ CUDA Convolution completed in " << milliseconds / 1000.0 << " seconds." << endl;

    return 0;
}