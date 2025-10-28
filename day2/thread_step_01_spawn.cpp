#include <thread>
#include <iostream>
using namespace std;
// Step 01: Launching and joining a single thread.
// Fill in each TODO so that the program prints the main thread id,
// starts a worker thread, and waits for it to finish.

void report_from_worker(int value) {
   
    // TODO: print a message containing the worker thread id and the value.
    cout<<"[worker value:]"<<value<<" thread id:"<< this_thread::get_id()<<"\n";
}

int main() {
    // TODO: print the main thread's id.
    cout<<this_thread::get_id();
    int va =13;
      
    // TODO: create a std::thread that calls report_from_worker with an int argument.
thread t1(report_from_worker,va);
thread t2(report_from_worker,va);
    // TODO: wait for the worker thread to finish by calling join().
    try{
        t1.join();
       
        t2.join();
    
        cout<<"hello";
    }
    catch(exception)
    {
        exception(e);
        t1.join();
        t2.join();
        t1.detach();
        t2.detach();

    }
    return 0;
}
