#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

// Goal: see how a condition_variable hands work from one thread to another.
namespace day5::cv_demo {
std::mutex mtx;
std::condition_variable cv;
bool workReady = false;

void producer() {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "producer prepared data\n";
        workReady = true;
    }
    cv.notify_one();
}

void consumer() {
    std::unique_lock<std::mutex> lock(mtx);
    //cv.wait(lock, [] { return workReady; });
    std::cout << "consumer picked up data\n";
    workReady = false; // reset for experiment
}
} // namespace day5::cv_demo

int main() {
    using namespace day5::cv_demo;

    std::cout << "Experiment: condition_variable handshake\n";
    std::thread tProducer(producer);
    std::thread tConsumer(consumer);

    tProducer.join();
    tConsumer.join();

    std::cout << "Both threads finished. Try changing sleep time or removing notify to observe deadlock." << std::endl;
    return 0;
}
