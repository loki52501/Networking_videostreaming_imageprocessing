#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

// Goal: practice using a queue guarded by mutex + condition_variable.
class SafeQueue {
public:
    void push(int value) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(value);
        }
        cv_.notify_one();
    }

    std::optional<int> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || stop_; })) {
            return std::nullopt; // timed out
        }
        if (queue_.empty()) {
            return std::nullopt; // woken because stop_ set
        }
        int value = queue_.front();
        queue_.pop();
        return value;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<int> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_ = false;
};

int main() {
    SafeQueue queue;
    std::atomic<bool> keepRunning{true};

    std::thread worker([&] {
        while (keepRunning.load()) {
            auto item = queue.pop(std::chrono::milliseconds(200));
            if (!item.has_value()) {
                if (!keepRunning.load()) break; // graceful stop
                std::cout << "worker idle...\n";
                continue;
            }
            std::cout << "worker got " << *item << "\n";
        }
        std::cout << "worker exiting\n";
    });

    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::cout << "main pushing " << i << "\n";
        queue.push(i);
    }

    keepRunning.store(false);
    queue.stop();
    worker.join();
    return 0;
}
