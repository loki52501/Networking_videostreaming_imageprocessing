#include <iostream>
#include <future>
#include <thread>
#include <functional>

template<class F, class... Args>
auto submit(F&& f, Args&&... args)
  ->std::invoke_result_t< F, Args...>
{
   using ret=std::invoke_result_t<F,Args...>
   
 return std::invoke(std::forward<F>(f),std::forward<Args>(args)...);
}
int main() {
    auto f1 = submit([](int a, int b) { return a + b; }, 2, 3);
std::cout<<f1<<"\n";
return -1;
}