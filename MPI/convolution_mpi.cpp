#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

typedef vector<vector<float>> Matrix;

// Clamp to keep pixel values in range
uchar clamp(float val) {
    return min(255.0f, max(0.0f, val));
}

// Apply convolution for given rows
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

            output.at<uchar>(i - startRow, j) = clamp(sum);
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
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };

    Mat inputImage;
    int rows = 0, cols = 0;

    if (rank == 0) {
        inputImage = imread("../Images/input.png", IMREAD_GRAYSCALE);
        if (inputImage.empty()) {
            cerr << "Error loading image!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = inputImage.rows;
        cols = inputImage.cols;
        cout << " Input image: " << rows << " x " << cols << endl;
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunkSize = rows / size;
    int extra = rows % size;

    int startRow = rank * chunkSize + min(rank, extra);
    int localRows = chunkSize + (rank < extra ? 1 : 0);

    // Add halo rows if not at image edge
    int haloTop = (startRow == 0) ? 0 : 1;
    int haloBottom = ((startRow + localRows) >= rows) ? 0 : 1;
    int totalRows = localRows + haloTop + haloBottom;

    Mat localInput(totalRows, cols, CV_8UC1);
    Mat localOutput(localRows, cols, CV_8UC1);

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
                    rowToSend = inputImage.row(0).clone();  // replicate top row
                else if (realRow >= rows)
                    rowToSend = inputImage.row(rows - 1).clone(); // replicate bottom row
                else
                    rowToSend = inputImage.row(realRow).clone();

                if (r == 0)
                    rowToSend.copyTo(localInput.row(i + hTop));
                else
                    MPI_Send(rowToSend.data, cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < totalRows; ++i) {
            MPI_Recv(localInput.ptr(i), cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Apply convolution only on the inner chunk (skip halo)
    applyConvolution(localInput, localOutput, kernel, haloTop, totalRows - haloBottom);

    if (rank == 0) {
        Mat fullResult(rows, cols, CV_8UC1);
        localOutput.copyTo(fullResult.rowRange(startRow, startRow + localRows));

        for (int r = 1; r < size; ++r) {
            int sRow = r * chunkSize + min(r, extra);
            int lRows = chunkSize + (r < extra ? 1 : 0);
            Mat temp(lRows, cols, CV_8UC1);
            MPI_Recv(temp.ptr(), lRows * cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            temp.copyTo(fullResult.rowRange(sRow, sRow + lRows));
        }

        imwrite("../Results/output_mpi.png", fullResult);

        double end_time = MPI_Wtime();
        cout << " MPI Convolution completed in " << (end_time - start_time) << " seconds." << endl;
    } else {
        MPI_Send(localOutput.ptr(), localRows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
