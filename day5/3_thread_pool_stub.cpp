#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t n) : stop(false) {
        for(size_t i = 0; i < n; i++) {
            workers.emplace_back([this]{
                while(true) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mu);
                        cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                        if(stop && tasks.empty()) return;
                        job = std::move(tasks.front());
                        tasks.pop();
                    }
                    job();
                }
            });
        }
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F,Args...>::type>
    {
        using Ret = typename std::invoke_result<F,Args...>::type;

        auto task = std::make_shared<std::packaged_task<Ret()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<Ret> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mu);
            tasks.emplace([task]{ (*task)(); });
        }
        cv.notify_one();
        return fut;
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv.notify_all();
        for(auto &t : workers) t.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mu;
    std::condition_variable cv;
    bool stop;
};

int main() {
    ThreadPool pool(6);

    // submit jobs and get futures
    auto f1 = pool.submit([]{
        long long s = 0;
        for(long long i = 0; i < 40'000'000; i++) s += i;
        return s;
    });

    auto f2 = pool.submit([]{
        long long s = 1;
        for(long long i = 1; i < 20'000'000; i++) s *= 1; // dummy
        return 42;
    });

    // block until result is ready
    std::cout << "Result1 = " << f1.get() << "\n";
    std::cout << "Result2 = " << f2.get() << "\n";

    return 0;
}
