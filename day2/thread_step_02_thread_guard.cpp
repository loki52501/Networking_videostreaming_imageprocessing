#include <chrono>
#include <iostream>
#include <thread>
using namespace std;
// Step 02: Resource Acquistion is Initialization RAII ownership of thread using a thread_guard.
// Implement the missing pieces so the guard always joins in its destructor.

class thread_guard {
    thread& thread_;

public:
    explicit thread_guard(thread& t);
    ~thread_guard();

    thread_guard(const thread_guard&) = delete;
    thread_guard& operator=(const thread_guard&) = delete;
};

void background_work();

int main() {
    cout << "[main] thread_guard demo start "<<this_thread::get_id()<<"\n";

    // TODO: launch background_work on a thread.
    thread t1(background_work);
    thread t22(background_work);
    // TODO: create a thread_guard that manages the new thread.
    thread_guard t2(t1);
    thread_guard t2a(t22);

    cout<< "[main] doing other work while background runs\n";
    this_thread::sleep_for(chrono::milliseconds(500));

    cout << "[main] leaving scope -> guard should join automatically\n";
    return 0;
}

// ---- Implementations ----

thread_guard::thread_guard(thread& t)
    : thread_(t) {
    // TODO: optionally validate thread state.

    cout<<"this is intialization "<<t.get_id()<<"\n";
}

thread_guard::~thread_guard() {
    // TODO: check if the thread is joinable and join it.
    if(this->thread_.joinable())
    {cout<<"this is the destructor \n";
        this->thread_.join();
    }}

void background_work() {
    for (int i = 0; i < 3; ++i) {
        cout << "[worker] cycle " << i
                  << " thread_id=" << this_thread::get_id() << '\n';
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}
