#include <iostream>
#include <vector>
#include<chrono>
using namespace std;
using namespace chrono;
int main() {
    const int n = 10000000;
    std::vector<float> A(n), B(n), C(n);
 steady_clock::time_point start=high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }

    for (float v : C);
    std::cout << '\n';
    cout<<" time taken: "<<duration<float,milli>(high_resolution_clock::now()-start).count()<<" ms \n";
    return 0;
}