#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

// TODO: 1) Change counter_t to std::int64_t or a custom struct once you grasp
// the basic atomic<int> behavior.
using counter_t = int;

// Shared atomic counter that multiple threads will update.
std::atomic<counter_t> counter{0};

// Demonstrates relaxed ordering increments.
void relaxed_increment(int increments_per_thread)
{
    for (int i = 0; i < increments_per_thread; ++i) {
        // TODO: Experiment with fetch_add memory orderings:
        // - std::memory_order_relaxed (baseline)
        // - std::memory_order_acq_rel
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}

// Demonstrates a compare/exchange loop with explicit success/failure orders.
void compare_exchange_spin(int increments_per_thread)
{
    for (int i = 0; i < increments_per_thread; ++i) {
        counter_t expected = counter.load(std::memory_order_relaxed);
        while (!counter.compare_exchange_weak(
            expected,
            expected + 1,
            std::memory_order_acq_rel,   // TODO: adjust success ordering.
            std::memory_order_relaxed)) // TODO: adjust failure ordering.
        {
            // TODO: add back-off (std::this_thread::yield) when experimenting
            // with high contention.
        }
    }
}

// Demonstrates an acquire/release pair for hand-off between threads.
void release_store_acquire_load(std::atomic<bool>& ready_flag)
{
    // Producer: set counter then publish.
    counter.store(42, std::memory_order_relaxed);
    ready_flag.store(true, std::memory_order_release); // TODO: flip to relaxed and observe data races.
}

void wait_for_ready(std::atomic<bool>& ready_flag)
{
    while (!ready_flag.load(std::memory_order_acquire)) {
        // TODO: add timeout or use wait()/notify_one() in C++20 core experiments.
    }
    std::cout << "Observed counter=" << counter.load(std::memory_order_relaxed) << '\n';
}

int main()
{
    constexpr int kThreads = 4;
    constexpr int kIncrementsPerThread = 100'000;

    // TODO: Reset counter to 0 before each experiment to isolate results.
    counter = 0;

    {
        std::vector<std::thread> workers;
        workers.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            workers.emplace_back(relaxed_increment, kIncrementsPerThread);
        }
        for (auto& t : workers) {
            t.join();
        }
        std::cout << "[Relaxed fetch_add] counter=" << counter.load() << '\n';
        // TODO: Run multiple times and record whether the result is deterministic.
    }

    counter = 0;
    {
        std::vector<std::thread> workers;
        workers.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            workers.emplace_back(compare_exchange_spin, kIncrementsPerThread);
        }
        for (auto& t : workers) {
            t.join();
        }
        std::cout << "[CAS loop] counter=" << counter.load() << '\n';
        // TODO: Profile contention cost vs. fetch_add.
    }

    counter = 0;
    {
        std::atomic<bool> ready_flag{false};
        std::thread producer(release_store_acquire_load, std::ref(ready_flag));
        std::thread consumer(wait_for_ready, std::ref(ready_flag));
        producer.join();
        consumer.join();
        // TODO: Deliberately comment out release/acquire ordering and observe stale reads.
    }

    return 0;
}

