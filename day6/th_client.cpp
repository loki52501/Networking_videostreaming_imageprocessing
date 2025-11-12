#include<iostream>
#include<thread>
#include<deque>
#include<future>
#include<vector>
#include<functional>
#include<memory>
#include<string>
#include<chrono>
#include "clientclass.h"
#include "serverclass.h"
using namespace std;
using namespace chrono;

class threadpool{
high_resolution_clock::time_point start=high_resolution_clock::now();
vector<thread> workers;
mutex mtx;
condition_variable cv;
deque<function<void()>> tasks;
bool stop;
public:
threadpool(size_t n) :stop(false)
{
    for(size_t i=0;i<n;i++)
    {
  workers.emplace_back([this]{
  while(true)
  {
    function<void()>job;
    {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock,[this]{ return stop || !tasks.empty();});
        if(stop && tasks.empty())
        return;
        job=move(tasks.front());
        tasks.pop_front();
    }
    job();
}
  });}
}

template<class F, class... Args>
auto submit(F&& f, Args&&... args)
->future<invoke_result_t<F,Args...>>
{
using Ret=invoke_result_t<F,Args...>;
auto task=make_shared<packaged_task<Ret()>>(std::bind(forward<F>(f),forward<Args>(args)...));
future<Ret>fut=task->get_future();
{
    lock_guard<mutex>lock(mtx);
    if(stop)throw runtime_error("pool stopped");
    tasks.emplace_back([task]{(*task)();});

}
cv.notify_one();
return fut;
}

~threadpool() {
        {
            lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for(auto &t : workers) t.join();
      cout<<" \n time take to complete the thread id: "<<this_thread::get_id()<<" is:"<<duration<float,milli>(high_resolution_clock::now()-start).count()<<" ms\n";

    }
};


int main()
{
threadpool T(3);

vector<future<void>>ars;
client<string> c;
for(int i=0;i<3;i++)
{ars.emplace_back( T.submit([&]{ c.client_start(to_string(8000+i)); }));}
for(int i=0;i<3;i++)
{
    ars[i].get();
}

return -1;
}