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
        cerr << "❌ Error loading image: " << input_file << endl;
        return -1;
    }

    int rows = img.rows, cols = img.cols;
    cout << "✅ Input image size: " << rows << " x " << cols << endl;

    size_t imgSize = rows * cols * sizeof(uchar);
    size_t kernelSize = 3 * 3 * sizeof(float);

    float h_kernel[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    uchar *d_input, *d_output;
    float *d_kernel;

    cudaError_t err;

    err = cudaMalloc((void**)&d_input, imgSize);
    if (err != cudaSuccess) {
        cerr << "❌ cudaMalloc d_input failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void**)&d_output, imgSize);
    if (err != cudaSuccess) {
        cerr << "❌ cudaMalloc d_output failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void**)&d_kernel, kernelSize);
    if (err != cudaSuccess) {
        cerr << "❌ cudaMalloc d_kernel failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cout << "✅ Launching CUDA kernel with grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y
         << ") blocks, block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ") threads\n";

    // Timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_kernel, rows, cols, 3);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "❌ CUDA kernel launch failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    Mat result(rows, cols, CV_8UC1);
    cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    bool ok = imwrite(output_file, result);
    if (!ok) {
        cerr << "❌ Failed to save output image to: " << output_file << endl;
        return -1;
    }

    cout << "✅ Output saved to: " << output_file << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    cout << "✅ CUDA Convolution completed in " << milliseconds / 1000.0 << " seconds." << endl;

    return 0;
}