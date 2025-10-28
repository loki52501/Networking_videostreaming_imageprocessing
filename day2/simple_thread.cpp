#include<iostream>
#include<thread>
#include <cstdint>
#include <iomanip>
#include <string>

using namespace std;


void print_fn_addr(void (*f)(), const char* name) {
    auto ss = reinterpret_cast<std::uintptr_t>(f);
cout << std::hex << std::showbase << ss    // prints in hex (e.g. 0x1234)
          << std::dec << ' ' << ss << '\n';  }


struct funco
{
    int s;
    funco(int so): s(so) {}

    void simp() {
        auto ptr = static_cast<const void*>("hello this is " + s);
auto ss  = reinterpret_cast<std::uintptr_t>(ptr);

cout << std::hex << std::showbase << ss    // prints in hex (e.g. 0x1234)
          << std::dec << ' ' << ss << '\n';  
    }

    void operator()() {      // callable body
        simp();              // do the work the thread should run
    }
};

void testingthread(int cro)
{
    funco lo(cro);
    cout<<this_thread::get_id()<<"\n";
    thread s1(lo);
    s1.join();

}
void hellow()
{
    cout<<" hello world from thread 2 "<<this_thread::get_id()<<"\n";
}
void hellow2()
{
    cout<<" hello world from thread 3 "<<this_thread::get_id()<<"\n";
}
void hellow3()
{
    cout<<" hello world from thread 4 "<<this_thread::get_id()<<"\n";
}


int main()
{
    print_fn_addr(&hellow,"hellow");
        print_fn_addr(&hellow2,"hellow2");

            print_fn_addr(&hellow3,"hellow3");


    thread t1(hellow);
    t1.join();
    thread t2(hellow2);
    thread t3(hellow3);
    thread t4(testingthread,3);
    t3.join();
    t2.detach();
  t4.detach();
    return 0;
}