#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
using namespace std;
// Step 06: Ring buffer skeleton ready for real capture/display threads.
// Complete the TODOs to implement a bounded queue with stop signalling.

class RingBuffer {
    private:
    size_t capacity_;
    deque<int> buffer_;
    bool stop_ = false;
    mutex mutex_;
    condition_variable cv_not_empty_;
    condition_variable cv_not_full_;
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity) {}

    void push(int value) {
        unique_lock<mutex> lock(mutex_);
        cv_not_full_.wait(lock, [&] {
            return (capacity_>buffer_.size())|| stop_; /* TODO: allow push when buffer has room OR stop requested */;
        });
        if (stop_) {
            return ;  // stop requested: ignore further pushes
        }

        // TODO: add value to the buffer.
        buffer_.push_back(value);
        cv_not_empty_.notify_one();
    }

    optional<int> pop() {
        unique_lock<mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [&] {
            return !buffer_.empty() || stop_;/* TODO: allow pop when buffer not empty OR stop requested */;
        });
        if (buffer_.empty()) {
            return nullopt;  // drained with stop signal
        }

        int value = 0;
        // TODO: remove the front item and store it in 'value'.
        value=buffer_.front();
        buffer_.pop_front();
        cv_not_full_.notify_one();
        return value;  // make sure 'value' holds what you popped
    }

    void request_stop() {
        {
            lock_guard<mutex> lock(mutex_);
            stop_ = true;
        }
        cv_not_full_.notify_all();
        cv_not_empty_.notify_all();
    }


};

void capture_simulator(RingBuffer& buffer) {
    for (int frame = 0; frame < 10; ++frame) {
        cout << "[capture] frame " << frame << " captured\n";
        // TODO: push frame into the buffer.
        buffer.push(frame);
        this_thread::sleep_for(chrono::milliseconds(30));
    }
    buffer.request_stop();
    cout << "[capture] stop requested\n";
}

void display_simulator(RingBuffer& buffer) {
    while (true) {
        // TODO: pop from buffer; break when nullopt is returned.
        optional<int> pr=buffer.pop();
        if(pr==nullopt)
        break;
        cout << "[display] frame "<<*pr<<" rendered\n";  // TODO: replace ??? with the popped value.
        this_thread::sleep_for(chrono::milliseconds(60));
    }
    cout << "[display] drained buffer and exiting\n";
}

int main() {
    cout << "[main] ring buffer skeleton demo\n";

    RingBuffer buffer(3);

    // TODO: launch capture_simulator and display_simulator on threads (use ref for buffer).
      thread t(capture_simulator,ref(buffer));
      thread t2(display_simulator,ref(buffer));
    // TODO: join both threads.
t.join();
t2.join();
    cout << "[main] threads joined\n";
    return 0;
}
