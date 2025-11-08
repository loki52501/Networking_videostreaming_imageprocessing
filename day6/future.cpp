#include <iostream>
#include <utility>
#include <vector>

void sink(const std::vector<int>& v) { std::cout << "lvalue, size = " << v.size() << "\n"; }
void sink(std::vector<int>&& v)      { std::cout << "rvalue, size = " << v.size() << "\n"; }

template<class T>
void relay(T&& arg) {
    sink(arg);                       // always lvalue → calls lvalue overload
    sink(std::forward<T>(arg));      // preserves original value category
}

int main() {
    std::vector<int> v{1,2,3};
    relay(v);                        // prints: lvalue ... / lvalue ...
    relay(std::vector<int>{4,5});    // prints: lvalue ... / rvalue ...
}