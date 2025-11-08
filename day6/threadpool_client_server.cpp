#include<iostream>
#include<thread>
#include<deque>
#include<future>
#include<vector>
#include<functional>
#include<memory>
#include "Matrixc.h"
using namespace std;


class threadpool{
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
    }
};


int main()
{
threadpool T(6);
Matrix A;
vector<future<void>>jobs;
cout<<" this is Matrix A:\n";
A.printes(A.A);
cout<<" this is Matrix B:\n";

A.printes(A.B);
for(int i=0;i<4;i++)
{
    for(int j=0;j<4;j++)
 jobs.push_back( T.submit([&,i,j]{A.matrixadd(A.A,A.B,i,j);}));
}
cout<<jobs.size();
for(int i=0;i<15;i++)
{
   jobs[i].get();}


cout<<"\n";
A.printes(A.C);



return -1;
}