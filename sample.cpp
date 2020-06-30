#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    Mat bgr_image = imread("/Users/srie/Documents/Upwork/Scott/Fingerprint_binary/finger_160x160.png");
    Mat hls_image;
    cvtColor(bgr_image, hls_image, COLOR_BGR2HLS);

    // Extract the L channel
    vector<cv::Mat> hls_planes(3);
    cv::split(hls_image, hls_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(3.0);
    clahe->setTilesGridSize(Size(4,4));
    cv::Mat dst;
    clahe->apply(hls_planes[1], dst);
    cout << "Original L = " << endl << " " << hls_planes[1] << endl << endl;
    cout << "Clahe L = " << endl << " " << dst << endl << endl;

//     // Merge the the color planes back into an Lab image
//     dst.copyTo(lab_planes[0]);
//     cv::merge(lab_planes, lab_image);

//    // convert back to RGB
//    cv::Mat image_clahe;
//    cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
    return 0;
}