#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

double computeRMSE(const Mat& img1, const Mat& img2) {
    CV_Assert(img1.rows == img2.rows && img1.cols == img2.cols);
    CV_Assert(img1.type() == img2.type());

    double mse = 0.0;

    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
            double diff = static_cast<double>(img1.at<uchar>(i, j)) - img2.at<uchar>(i, j);
            mse += diff * diff;
        }
    }

    mse /= (img1.rows * img1.cols);
    return sqrt(mse);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: ./compare_rmse img1.png img2.png" << endl;
        return -1;
    }

    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "❌ Error loading images!" << endl;
        return -1;
    }

    double rmse = computeRMSE(img1, img2);
    cout << "✅ RMSE between " << argv[1] << " and " << argv[2] << ": " << rmse << endl;

    return 0;
}
