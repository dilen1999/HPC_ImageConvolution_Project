#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

// CUDA clamp helper
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

// CUDA launcher
void applyConvolutionCUDA(const Mat& input, Mat& output, vector<vector<float>>& kernel, int rank, const string& filterName) {
    int rows = input.rows;
    int cols = input.cols;
    int kSize = kernel.size();
    size_t imgSize = rows * cols * sizeof(uchar);
    size_t kernelSize = kSize * kSize * sizeof(float);

    uchar *d_input, *d_output;
    float *d_kernel;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_kernel, kernelSize);

    cudaMemcpy(d_input, input.data, imgSize, cudaMemcpyHostToDevice);

    float h_kernel[kSize * kSize];
    for (int i = 0; i < kSize; ++i)
        for (int j = 0; j < kSize; ++j)
            h_kernel[i * kSize + j] = kernel[i][j];

    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaConvolution<<<blocks, threads>>>(d_input, d_output, d_kernel, rows, cols, kSize);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf(" Rank %d: CUDA %s filter took %.6f seconds\n", rank, filterName.c_str(), milliseconds / 1000.0f);

    cudaMemcpy(output.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    Mat inputImage;
    int rows, cols;

    if (rank == 0) {
        inputImage = imread("../Images/input.png", IMREAD_GRAYSCALE);
        if (inputImage.empty()) {
            cerr << " Error loading image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = inputImage.rows;
        cols = inputImage.cols;
        cout << " Input: " << rows << " x " << cols << endl;
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunkSize = rows / size;
    int extra = rows % size;
    int startRow = rank * chunkSize + min(rank, extra);
    int localRows = chunkSize + (rank < extra ? 1 : 0);

    int haloTop = (startRow == 0) ? 0 : 1;
    int haloBottom = ((startRow + localRows) >= rows) ? 0 : 1;
    int totalRows = localRows + haloTop + haloBottom;

    Mat localInput(totalRows, cols, CV_8UC1);

    // Scatter
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int sRow = r * chunkSize + min(r, extra);
            int lRows = chunkSize + (r < extra ? 1 : 0);
            int hTop = (sRow == 0) ? 0 : 1;
            int hBottom = ((sRow + lRows) >= rows) ? 0 : 1;

            for (int i = -hTop; i < lRows + hBottom; ++i) {
                int realRow = sRow + i;
                Mat rowToSend;
                if (realRow < 0)
                    rowToSend = inputImage.row(0).clone();
                else if (realRow >= rows)
                    rowToSend = inputImage.row(rows - 1).clone();
                else
                    rowToSend = inputImage.row(realRow).clone();

                if (r == 0)
                    rowToSend.copyTo(localInput.row(i + hTop));
                else
                    MPI_Send(rowToSend.data, cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < totalRows; ++i)
            MPI_Recv(localInput.ptr(i), cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Extract core submatrix
    Mat localChunk = localInput.rowRange(haloTop, totalRows - haloBottom).clone();

    // Kernels
    vector<vector<float>> sharpenKernel = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    vector<vector<float>> blurKernel = {{1.0/9, 1.0/9, 1.0/9}, {1.0/9, 1.0/9, 1.0/9}, {1.0/9, 1.0/9, 1.0/9}};
    vector<vector<float>> edgeKernel = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

    Mat localSharpen(localRows, cols, CV_8UC1);
    Mat localBlur(localRows, cols, CV_8UC1);
    Mat localEdge(localRows, cols, CV_8UC1);

    applyConvolutionCUDA(localChunk, localSharpen, sharpenKernel, rank, "Sharpen");
    applyConvolutionCUDA(localChunk, localBlur, blurKernel, rank, "Blur");
    applyConvolutionCUDA(localChunk, localEdge, edgeKernel, rank, "Edge");

    vector<pair<Mat*, string>> outputs;
    outputs.push_back(make_pair(&localSharpen, "sharpen"));
    outputs.push_back(make_pair(&localBlur, "blur"));
    outputs.push_back(make_pair(&localEdge, "edge"));

    for (size_t i = 0; i < outputs.size(); ++i) {
        Mat* localOutput = outputs[i].first;
        string name = outputs[i].second;

        if (rank == 0) {
            Mat fullResult(rows, cols, CV_8UC1);
            localOutput->copyTo(fullResult.rowRange(startRow, startRow + localRows));
            for (int r = 1; r < size; ++r) {
                int sRow = r * chunkSize + min(r, extra);
                int lRows = chunkSize + (r < extra ? 1 : 0);
                Mat temp(lRows, cols, CV_8UC1);
                MPI_Recv(temp.ptr(), lRows * cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                temp.copyTo(fullResult.rowRange(sRow, sRow + lRows));
            }
            string filename = "../Results/output_hybrid_" + name + ".png";
            imwrite(filename, fullResult);
            cout << " Saved: " << filename << endl;
        } else {
            MPI_Send(localOutput->ptr(), localRows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        cout << " Total MPI + CUDA Hybrid time: " << (end_time - start_time) << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
