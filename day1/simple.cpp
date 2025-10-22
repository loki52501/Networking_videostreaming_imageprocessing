#include <opencv2/opencv.hpp>
int main(){ cv::Mat img(100,100,CV_8UC3,cv::Scalar(0,255,0)); cv::imshow("test",img); cv::waitKey(0);}
