// Day 1 — "Hello, Memory" Video Reader with FPS and timing
// Reads from default camera (0) or a file if a path is provided.
// Displays grayscale video, instantaneous FPS, and processing time.
#include <iostream>
#include <string>
#include <vector>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;


static void printFrameMemoryInfo(const Mat &frame) {
    if (frame.empty()) return;
    int width = frame.cols;
    int height = frame.rows;
    int channels = frame.channels();
    size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
    cout << "Frame: " << width << "x" << height
         << " x " << channels << " bytes/channel = " << bytes << " bytes (~"
         << (bytes / (1024.0 * 1024.0)) << " MB)" << endl;
}

int main(int argc, char** argv) {
    // Input source: default camera or file path passed as first argument
    bool useCamera = true;
   string sourcePath;
    if (argc > 1) {
        useCamera = false;
        sourcePath = argv[1];
    }

    VideoCapture cap;
    if (useCamera) {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "Failed to open default camera (0). "
                      << "Tip: pass a video file path as an argument, e.g., gray_video <file.mp4>\n";
            return -1;
        }
    } else {
        cap.open(sourcePath);
        if (!cap.isOpened()) {
            cerr << "Failed to open source: " << sourcePath << "\n";
            return -1;
        }
    }

    // Optionally set a reasonable resolution to avoid huge frames on some webcams
    if (useCamera) {
        cap.set(CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    }

    Mat frame, gray;
    const double freq = getTickFrequency();
    int64 prevTick = getTickCount();
    int64 frameStartTick = 0;
    double procMs = 0.0;
    double procMsAvg = 0.0;
    const double alpha = 0.05; // EMA smoothing factor for average processing time

    // Warm up: read first valid frame to report memory info
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "No frames from source." << std::endl;
        return -1;
    }
    printFrameMemoryInfo(frame);

    // Pre-allocate gray with same size (single channel)
    gray.create(frame.size(), CV_8UC1);
    const std::string windowName = "Gray Video";
    namedWindow(windowName, WINDOW_NORMAL);

    // Show the first frame as grayscale too
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    imshow(windowName, gray);
    if (waitKey(1) == 27) return 0; // ESC

    while (true) {
        frameStartTick = getTickCount();

        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Stream ended or frame empty." << std::endl;
            break;
        }

        // Processing: BGR -> Gray
        int64 t0 = getTickCount();
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        int64 t1 = getTickCount();
        procMs = (t1 - t0) * 1000.0 / getTickFrequency();
        procMsAvg = (1.0 - alpha) * procMsAvg + alpha * procMs; // exponential moving average

        // Compute instantaneous FPS based on wall time between frames
        int64 now = getTickCount();
        double dt = (now - prevTick) / getTickFrequency(); // seconds
        prevTick = now;
        double fps = (dt > 0.0) ? (1.0 / dt) : 0.0;

        // Display with live stats
        std::string title = windowName + " | FPS: " + cv::format("%.1f", fps)
                           + " | proc: " + cv::format("%.2f", procMs) + " ms"
                           + " (avg " + cv::format("%.2f", procMsAvg) + " ms)";
        setWindowTitle(windowName, title);
        imshow(windowName, gray);

        // Exit on ESC
        int key = waitKey(1);
        if (key == 27) break;
    }

    return 0;
}

