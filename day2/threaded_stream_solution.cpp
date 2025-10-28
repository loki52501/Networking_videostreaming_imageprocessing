// Day 2 Solution — threaded_stream_solution.cpp
// Capture -> Process -> Display pipeline with a ring buffer between each stage.

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <iostream>
#include <atomic>

using namespace cv;
using namespace std;

template<typename T>
class RingBuffer {
public:
    RingBuffer(size_t capacity): buf(capacity), cap(capacity), head(0), tail(0), count(0) {}

    void push(const T& item) {
        unique_lock<mutex> lock(mux);
        not_full.wait(lock, [this]{ return count < cap || stop.load(); });
        if (stop.load()) return;
        buf[head] = item.clone();
        head = (head + 1) % cap;
        ++count;
        lock.unlock();
        not_empty.notify_one();
    }

    bool pop(T& out) {
        unique_lock<mutex> lock(mux);
        not_empty.wait(lock, [this]{ return count > 0 || stop.load(); });
        if (count == 0) return false;
        out = buf[tail].clone();
        tail = (tail + 1) % cap;
        --count;
        lock.unlock();
        not_full.notify_one();
        return true;
    }

    void shutdown() {
        stop = true;
        not_empty.notify_all();
        not_full.notify_all();
    }

private:
    vector<T> buf;
    size_t cap;
    size_t head, tail;
    size_t count;
    mutex mux;
    condition_variable not_empty, not_full;
    atomic<bool> stop{false};
};

int main() {
    const size_t BUF_SIZE = 3;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera\n";
        return -1;
    }

    RingBuffer<Mat> cap_proc(BUF_SIZE);
    RingBuffer<Mat> proc_disp(BUF_SIZE);

    atomic<bool> done{false};

    // Capture thread
    thread tcap([&]{
        while (!done) {
            Mat frame;
            if (!cap.read(frame)) { done = true; break; }
            cap_proc.push(frame);
        }
        cap_proc.shutdown();
    });

    // Process thread (grayscale)
    thread tproc([&]{
        Mat in;
        while (cap_proc.pop(in)) {
            Mat gray;
            cvtColor(in, gray, COLOR_BGR2GRAY);
            cvtColor(gray, gray, COLOR_GRAY2BGR); // convert back for display consistency
            proc_disp.push(gray);
        }
        proc_disp.shutdown();
    });

    // Display thread
    thread tdisp([&]{
        Mat frame;
        while (proc_disp.pop(frame)) {
            if (frame.empty()) continue;
            imshow("Threaded Stream (Solution)", frame);
            if (waitKey(1) == 27) { done = true; break; }
        }
    });

    tcap.join();
    tproc.join();
    tdisp.join();
    return 0;
}
