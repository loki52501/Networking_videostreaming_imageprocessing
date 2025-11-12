#include <iostream>
#include <utility>
#include <vector>
using namespace std;
// void sink(const std::vector<int> &v)
// {

//     std::cout << "lvalue, size = " << v.size() << "\n";
// }
// void sink(std::vector<int> &&v) { std::cout << "rvalue, size = " << v.size() << "\n"; }

// template <class T>
// void relay(T &&arg)
// {
//     sink(arg);                  // always lvalue → calls lvalue overload
//     sink(std::forward<T>(arg)); // preserves original value category
// }

// int main()
// {
//     std::vector<int> v{1, 2, 3};
//     relay(v);                      // prints: lvalue ... / lvalue ...
//     relay(std::vector<int>{4, 5}); // prints: lvalue ... / rvalue ...
// }
void takes_lvalue(int &x) { x=x+10; cout<<x; }
void takes_rvalue(int&& x) { x=x+10;cout<<x; }

int main() {
    int a = 10;int value = 42;
    
    int* p = &value;     // p is an lvalue (named object), it points to value
    int* q = new int(7); // q is an lvalue; the pointed-to int is elsewhere
       int *&ref=p;
       ref=ref+10;
       cout<<ref<< " older , this is newer: "<<p;
       int *&&reff=new int(10);
       reff+=10;
       cout<<p<<" old value : new Value: "<<reff;
}