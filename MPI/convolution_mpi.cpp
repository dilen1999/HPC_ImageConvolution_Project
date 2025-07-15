#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

typedef vector<vector<float>> Matrix;

uchar clamp(float val) {
    return min(255.0f, max(0.0f, val));
}

void applyConvolution(const Mat& input, Mat& output, const Matrix& kernel, int startRow, int endRow) {
    int kSize = kernel.size();
    int offset = kSize / 2;

    for (int i = startRow; i < endRow; ++i) {
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
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    Matrix kernel = {
        {0, -1,  0},
        {-1, 5, -1},
        {0, -1,  0}
    };

    Mat inputImage, fullResult;
    int rows, cols;

    if (rank == 0) {
        inputImage = imread("../Images/input.png", IMREAD_GRAYSCALE);
        if (inputImage.empty()) {
            cerr << "Error loading image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = inputImage.rows;
        cols = inputImage.cols;
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunkSize = rows / size;
    int extra = rows % size;
    int startRow = rank * chunkSize + min(rank, extra);
    int localRows = chunkSize + (rank < extra ? 1 : 0);
    Mat localInput(localRows + 2, cols, CV_8UC1); // include halo rows
    Mat localOutput(localRows, cols, CV_8UC1);

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int sRow = r * chunkSize + min(r, extra);
            int rowsToSend = chunkSize + (r < extra ? 1 : 0);
            for (int i = -1; i <= rowsToSend; ++i) {
                int realRow = sRow + i;
                Mat rowToSend = (realRow >= 0 && realRow < rows) ? inputImage.row(realRow) : Mat::zeros(1, cols, CV_8UC1);
                if (r == 0)
                    rowToSend.copyTo(localInput.row(i + 1));
                else
                    MPI_Send(rowToSend.data, cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < localRows + 2; ++i) {
            MPI_Recv(localInput.ptr(i), cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    applyConvolution(localInput, localOutput, kernel, 1, localRows + 1);

    if (rank == 0) {
        fullResult = Mat(rows, cols, CV_8UC1);
        localOutput.copyTo(fullResult.rowRange(startRow, startRow + localRows));
        for (int r = 1; r < size; ++r) {
            int sRow = r * chunkSize + min(r, extra);
            int rowsToRecv = chunkSize + (r < extra ? 1 : 0);
            Mat temp(rowsToRecv, cols, CV_8UC1);
            MPI_Recv(temp.data, rowsToRecv * cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            temp.copyTo(fullResult.rowRange(sRow, sRow + rowsToRecv));
        }
        imwrite("../Results/output_mpi.png", fullResult);
        double end_time = MPI_Wtime();
        cout << "✅ MPI Convolution completed in " << (end_time - start_time) << " seconds." << endl;
    } else {
        MPI_Send(localOutput.data, localRows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}