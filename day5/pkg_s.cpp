#include<iostream>
#include<thread>
#include<deque>
#include<future>
#include"timing_matrix_ops.hpp"
using namespace timing;
using namespace std;
using namespace chrono;
using Matrix = std::vector<std::vector<int>>;

class Threadpoolmatrix{
Matrix A,B,C;
mutex mtx;
deque<packaged_task<Matrix(Matrix&, Matrix&)>> tasks;
high_resolution_clock::time_point start =high_resolution_clock::now();
public:
Threadpoolmatrix(int row, int column,int row2, int column2)
{A=Matrix(row,vector<int>(column,1));
 B=Matrix(row2,vector<int>(column2,1));   
    randomvalue(A,row, column);
    randomvalue(B,row2, column2);
}
future<Matrix> start_addthread()
{auto startadd=high_resolution_clock::now();
cout<<"\n the matrix addition started at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
 packaged_task<Matrix(Matrix&,Matrix&)> task(matrixadd);
future<Matrix> res=task.get_future();
lock_guard<mutex>lk(mtx);
tasks.push_back(move(task));
return res;
}
future<Matrix> start_subthread()
{auto startadd=high_resolution_clock::now();
cout<<"\n the matrix subtraction at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
 packaged_task<Matrix(Matrix&,Matrix&)> task(matrixsub);
future<Matrix> res=task.get_future();
lock_guard<mutex>lk(mtx);
tasks.push_back(move(task));
return res;
}
future<Matrix> start_multhread()
{auto startadd=high_resolution_clock::now();
cout<<"\n the matrix multiplication time started at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
 packaged_task<Matrix(Matrix&,Matrix&)>task(matrixmul);
future<Matrix> res=task.get_future();
lock_guard<mutex>lk(mtx);
tasks.push_back(move(task));

return res;
}
void main_thread()
{   auto startadd=high_resolution_clock::now();
cout<<"\n the main thread time started at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
    while(!tasks.empty())
    {packaged_task<Matrix(Matrix&,Matrix&)> task;
    {
       lock_guard<mutex>lk(mtx);
        if(tasks.empty())
        return;
        task=move(tasks.front());
        tasks.pop_front();
    }
    task(A,B);}
}
void add_thread()
{
 start_addthread();
    start_multhread();
  start_subthread();
    
}
void printMatrix(Matrix A)
{
    printes(A);
}
void printCurrMatrix()
{cout<<"\n this is matrix A:\n";
    printMatrix(A);
cout<<"\n this is matrix B:\n";
    printMatrix(B);
}
};

int main()
{
    int row=3, column=3;
    Threadpoolmatrix m(2,2,2,2);
    m.printCurrMatrix();
     m.add_thread();
     m.main_thread();
    return -1;
}
