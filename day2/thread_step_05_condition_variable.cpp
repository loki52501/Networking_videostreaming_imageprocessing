#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
using namespace std;
// Step 05: Producer/consumer with condition_variable.
// Implement the TODOs so the producer pushes integers and the consumer processes them.

mutex mtx;
condition_variable cv;
queue<int> work_items;
bool done = false;

void producer() {
        cout<<"producer lala"<<this_thread::get_id()<<"\n";

    for (int value = 1; value <= 5; ++value) {
        {
            lock_guard<mutex> l(mtx);
            work_items.push(value);
            // TODO: lock the mutex and push a value into the queue.
        }
        // TODO: notify the consumer that new work is available.
       cv.notify_one();
        this_thread::sleep_for(chrono::milliseconds(80));
    }

    {lock_guard<mutex> l(mtx);
        done=true;// TODO: lock the mutex and set 'done' to true.
    }
//cv.notify_all();// TODO: wake the consumer so it can exit.
}

void consumer() {
    cout<<"consumer lala"<<this_thread::get_id()<<"\n";
    while (true) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [&] { return !work_items.empty() || done; /* TODO: queue not empty OR done */ });

        if (!work_items.empty()) {
            int next=work_items.front();
            work_items.pop();// TODO: pop an item and process it outside the critical section.
            lock.unlock();

            cout << "[consumer] processing " << next << "\n";
        } else if (done) {
            cout << "[consumer] done signal received\n";
            break;
        }
    }
}

int main() {
    cout << "[main] condition_variable demo\n";

    // TODO: start producer and consumer threads.
    thread produc(producer);
    thread consu(consumer);
    // TODO: join both threads.
produc.join();
consu.join();
    cout << "[main] all threads joined\n";
    return 0;
}
