#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <numeric>
#include <mutex>
#include <thread>
#include <vector>
using namespace std;
// Scenario: Split a CPU-heavy range sum into partitions and expose a future
// so the caller can poll, block, or time out.

vector<int> make_data(size_t count)
{
    vector<int> data(count);
    iota(data.begin(), data.end(), 1);
    return data;
}

class simple_thread_pool
{
public:
    explicit simple_thread_pool(size_t thread_count)
    {
        workers_.reserve(thread_count);
        for (size_t i = 0; i < thread_count; ++i)
        {
            workers_.emplace_back([this]() { this->worker_loop(); });
        }
    }

    ~simple_thread_pool()
    {
        {
            unique_lock lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    void enqueue(function<void()> task)
    {
        {
            unique_lock lock(mutex_);
            tasks_.emplace_back(move(task));
        }
        condition_.notify_one();
    }

private:
    void worker_loop()
    {
        while (true)
        {
            function<void()> task;
            {
                unique_lock lock(mutex_);
                condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty())
                {
                    return;
                }
                task = move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    vector<thread> workers_;
    deque<function<void()>> tasks_;
    mutex mutex_;
    condition_variable condition_;
    bool stop_{false};
};

int accumulate_slice(const vector<int>& data, size_t begin, size_t end, const atomic<bool>* cancel_flag)
{
    // Emulate workload variance so wait_for experiments have meaning.
    this_thread::sleep_for(chrono::milliseconds(10));
    if (!cancel_flag)
    {
        return accumulate(data.begin() + begin, data.begin() + end, 0);
    }

    const bool is_cancelled = cancel_flag->load(memory_order_relaxed);
    if (is_cancelled)
    {
        return 0;
    }

    const int* start = data.data() + begin;
    const int* finish = data.data() + end;
    int total = 0;
    while (start != finish)
    {
        if (cancel_flag->load(memory_order_relaxed))
        {
            break;
        }
        total += *start;
        ++start;
    }
    return total;
}

int main()
{
    constexpr size_t kTotalItems = 1000;
    constexpr size_t kWorkers = 4;

    simple_thread_pool pool(kWorkers);
    auto data = make_data(kTotalItems);
    atomic<bool> cancel_requested{false};
    packaged_task<int()> task([&data, &pool, kWorkers, &cancel_requested]() {
        if (data.empty())
        {
            return 0;
        }

        const size_t partitions = [&]() {
            size_t worker_count = kWorkers == 0 ? 1 : kWorkers;
            return worker_count < data.size() ? worker_count : data.size();
        }();

        vector<future<int>> partial_results;
        partial_results.reserve(partitions);

        size_t base_size = data.size() / partitions;
        size_t remainder = data.size() % partitions;
        size_t begin = 0;
        for (size_t i = 0; i < partitions; ++i)
        {
            size_t slice = base_size + (i < remainder ? 1 : 0);
            size_t end = begin + slice;

            packaged_task<int()> slice_task([&data, begin, end, &cancel_requested]() {
                return accumulate_slice(data, begin, end, &cancel_requested);
            });
            future<int> slice_future = slice_task.get_future();
            pool.enqueue([task = move(slice_task)]() mutable { task(); });
            partial_results.emplace_back(move(slice_future));
            begin = end;
        }

        int total = 0;
        for (auto& partial : partial_results)
        {
            total += partial.get();
        }
        return total;
    });

    future<int> result = task.get_future();

    pool.enqueue([task = move(task)]() mutable { task(); });

    const auto poll_interval = chrono::milliseconds(20);
    while (true)
    {
        future_status status = result.wait_for(poll_interval);
        if (status == future_status::ready)
        {
            break;
        }
        if (status == future_status::timeout)
        {
            cout << "[future_walkthrough] status=timeout\n";
            continue;
        }
        if (status == future_status::deferred)
        {
            cout << "[future_walkthrough] status=deferred\n";
            result.wait();
            break;
        }
    }
    cout << "[future_walkthrough] total=" << result.get() << '\n';

    return 0;
}
