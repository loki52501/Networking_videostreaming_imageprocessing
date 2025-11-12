#include<iostream>
#include<thread>
#include<deque>
#include<future>
#include<vector>
#include<functional>
#include<memory>
#include<string>
#include<chrono>
#include<winsock2.h>
#include<windows.h>
#include "clientclass.h"
#include "serverclass.h"
using namespace std;
using namespace chrono;

struct WsaLifetime {
    WsaLifetime() { WSAStartup(MAKEWORD(2,2), &wsd); }
    ~WsaLifetime() { WSACleanup(); }
private:
    WSADATA wsd{};
};

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
auto task=make_shared<packaged_task<Ret()>>(bind(forward<F>(f),forward<Args>(args)...));
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
            lock_guard<mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for(auto &t : workers) t.join();
      cout<<" \n time take to complete the thread id: "<<this_thread::get_id()<<" is:"<<duration<float,milli>(high_resolution_clock::now()-start).count()<<" ms\n";

    }
};


int main() {
       WsaLifetime winsock;
    server<string> s0("8000");
    server<string> s1("8001");
    server<string> s2("8002");


    atomic<bool> running{true};
                     // RAII wrapper that calls WSAStartup/WSACleanup once
threadpool serverPool(3);
    // pool dedicated to long-lived accept loops

    threadpool workerPool(8);            // add this
    
    auto srvLoop = [&](server<string>& srv) {
                  srv.listening();  // bind + listen once
        while (running.load()) {   
          SOCKET client=srv.client_socket_creation();         // running is an atomic<bool> you control
            workerPool.submit([&srv,&client]{srv.handle_client(client) ; });
        }
        srv.shutdowns();                         // close listening socket when shutting down
    };

    vector<future<void>> serverFutures;
    serverFutures.emplace_back(serverPool.submit(srvLoop, ref(s0)));
    serverFutures.emplace_back(serverPool.submit(srvLoop, ref(s1)));
    serverFutures.emplace_back(serverPool.submit(srvLoop, ref(s2)));

    // separate pool for short-lived clients or per-connection work
    threadpool workerPoolc(8);
    workerPoolc.submit([]{ client<string> c; c.client_start("8000"); });
    workerPoolc.submit([]{ client<string> c; c.client_start("8001"); });
    workerPoolc.submit([]{ client<string> c; c.client_start("8002"); });

    // later, signal shutdown and wait
running.store(false);
s0.shutdowns();
s1.shutdowns();
s2.shutdowns();
for (auto& fut : serverFutures) fut.get();

}
