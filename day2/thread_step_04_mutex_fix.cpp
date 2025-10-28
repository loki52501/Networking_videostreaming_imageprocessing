#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
using namespace std;
// Step 04: Fix the data race from step 03 using mutex.

int counter_safe = 0;
mutex counter_mutex;

void increment_with_lock(int iterations) {
    // TODO: acquire the mutex (e.g., lock_guard) before touching counter_safe.
 lock_guard<mutex> guard(counter_mutex);
    
    // TODO: loop 'iterations' times and increment counter_safe while holding the lock.
    for(int i=0;i<iterations;i++)
    cout<<counter_safe++<<" "<<this_thread::get_id()<<"\n";
}

int main() {
    cout << "[main] starting mutex-protected increment demo\n";

    constexpr int thread_count = 4;
    constexpr int iterations = 2;
    vector<thread> t;

    // TODO: spawn threads running increment_with_lock.
    for(int i=0;i<thread_count;i++)
t.insert(t.begin(),thread(increment_with_lock,iterations));
    // TODO: join all threads.
    for(int i=0;i<thread_count;i++)
    t[i].join();

    const int expected = thread_count * iterations;
    cout << "[result] counter_safe=" << counter_safe
              << " expected=" << expected
              << " (mutex removes data race)\n";

    return 0;
}
