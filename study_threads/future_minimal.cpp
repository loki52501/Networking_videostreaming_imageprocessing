#include <future>
#include <iostream>
#include <thread>
using namespace std;
// Computes factorial asynchronously using packaged_task/future.
uint64_t factorial(uint32_t n)
{
    uint64_t result = 1;
    for (uint32_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

int main()
{
    packaged_task<uint64_t(uint32_t)> task(factorial);
    future<uint64_t> result = task.get_future();

    thread worker(move(task), 4);

    cout << "4! = " << result.get() << '\n';

    worker.join();
    return 0;
}

