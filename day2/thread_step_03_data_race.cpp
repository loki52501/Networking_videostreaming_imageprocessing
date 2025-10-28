#include <iostream>
#include <thread>
#include <vector>
using namespace std;
// Step 03: Observe a data race by incrementing shared state without synchronization.
// After filling the TODOs, run the program multiple times to see inconsistent results.

int counter = 0;  // shared state (intentionally unsynchronized)

void increment_many_times(int iterations) {
    // TODO: loop 'iterations' times and increment the shared counter.
    for(int i=0;i<iterations;i++)
    {cout<<counter++;
    }
}

int main() {
    cout << "[main] starting unsynchronized increment demo\n";

    constexpr int thread_count = 4;
    constexpr int iterations = 500000;
vector<thread>t;
    // TODO: launch 'thread_count' threads running increment_many_times.
    for(int i=0;i<4;i++)
{
    
    t.insert(t.begin(),thread(increment_many_times,iterations));
}
for(int i=0;i<4;i++)
{
    t[i].join();
}
    // TODO: join all threads.

    const int expected = thread_count * iterations;
    cout << "[result] counter=" << counter
              << " expected=" << expected
              << " (mismatch shows undefined behavior from a data race)\n";

    return 0;
}
