#include <opencv2/opencv.hpp>
#include <iostream>
#include<chrono>
using namespace std;
using namespace cv;
using namespace chrono;
// Day 1 – Gray Video capture template.
// Fill each TODO to connect the capture-to-display path and instrument FPS plus memory touchpoints.
int main() {
    // TODO: create a cv::VideoCapture bound to the default camera (index 0).
    VideoCapture v(0);
    // TODO: validate that the capture opened successfully; print an error to std::cerr and return -1 if not.
    if(!v.isOpened())
   { cerr<<"there is no webcam";
    return -1; } 
    v.set(CAP_PROP_FRAME_WIDTH, 1920);
    v.set(CAP_PROP_FRAME_HEIGHT,1080);
    //  declare cv::Mat buffers to hold the raw BGR frame and the grayscale conversion.
    Mat frame, grey;

    auto starttime=high_resolution_clock::now();
    int frame_count=0;
    double fps=0;
    while (true) {
      
        v >> frame;
        if(frame.empty())
        break;   
  
        cout<<frame.channels()<<"\n";
        cvtColor(frame,grey,COLOR_BGR2RGB);
   
        ostringstream title;
        title<<"gray frame--"<<fps<<" FPS";
        setWindowTitle("GRAY frame",title.str());
        imshow("GRAY frame",grey);
        frame_count++;
        auto endtime=high_resolution_clock::now();
            duration<double> dur=endtime-starttime;
        if(dur.count()>=1.0)
        {
            fps=frame_count/dur.count();
            cout<<dur.count()<<" dc fc "<<frame_count<<"\n";
            std::size_t bytes = grey.total() * grey.elemSize();  // ~921,600 bytes
std::cout << "Gray frame: " << grey.cols << "x" << grey.rows
          << " -> " << bytes / (1024.0 * 1024.0) << " MB\n";

            starttime=high_resolution_clock::now();
            frame_count=0;
        }
        std::cout << "FPS: " << fps
          << " | frame: " << static_cast<void*>(frame.data)
          << " | gray: " << static_cast<void*>(grey.data) << '\n';

        // TODO: exit the loop when the user presses ESC (waitKey(1) == 27).
        if(waitKey(1)==27)
        break;
    }

    // TODO: call cv::destroyAllWindows() (or an equivalent) before returning.
    destroyAllWindows();
    return 0;
}
