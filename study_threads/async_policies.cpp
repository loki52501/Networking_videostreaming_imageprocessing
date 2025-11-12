#include <chrono>
#include <future>
#include <iostream>
#include <optional>
#include <string>

using clock_t = std::chrono::steady_clock;

std::string heavy_task(std::string name, std::chrono::milliseconds duration)
{
    auto start = clock_t::now();
    std::this_thread::sleep_for(duration);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - start);
    return name + " finished in " + std::to_string(elapsed.count()) + "ms";
}

// Helper to time how long std::async deferred execution waits before running.
template <typename Future>
void report_future(std::string_view label, Future& future)
{
    auto start = clock_t::now();
    auto value = future.get();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - start);
    std::cout << "[" << label << "] " << value << " | waited " << elapsed.count() << "ms\n";
}

int main()
{
    std::chrono::milliseconds duration{250};

    // Case 1: Force asynchronous launch.
    auto async_future = std::async(std::launch::async, heavy_task, "async task", duration);

    // Case 2: Force deferred launch (runs lazily on get()).
    auto deferred_future = std::async(std::launch::deferred, heavy_task, "deferred task", duration);

    // Case 3: Allow implementation to choose; observe behavior on your platform.
    auto auto_future = std::async(std::launch::async | std::launch::deferred, heavy_task, "auto task", duration);

    // TODO: Insert wait_for polling before get() to identify when each future becomes ready.
    report_future("async", async_future);
    report_future("deferred", deferred_future);
    report_future("auto", auto_future);

    // TODO: Run with multiple async tasks and measure thread count (std::thread::hardware_concurrency).
    // TODO: Observe exception handling by throwing from heavy_task.
    return 0;
}

