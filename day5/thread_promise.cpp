#include <future>
#include <iostream>
#include <thread>
using namespace std;
void add_async(int a, int b, promise<int> prom) {
    prom.set_value(a + b);
}

int main() {
    promise<int> prom;
    future<int> fut = prom.get_future();

    thread worker(add_async, 7, 5, move(prom));
    cout << "7 + 5 = " << fut.get() << "\n";
    //worker.join();
}