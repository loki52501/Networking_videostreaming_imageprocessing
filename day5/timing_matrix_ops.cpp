

#include "timing_matrix_ops.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include<random>
using namespace std;
using namespace chrono;
namespace timing {
namespace {
    auto start = std::chrono::high_resolution_clock::now();
}
void printes(vector<vector<int>>&A)
{if(A.size()!=0)
    for(int i=0;i<A.size();i++)
    {for(int j=0;j<A[0].size();j++)
    {
        cout<<"|"<<A[i][j]<<" ";
    }
    cout<<"|\n";
}
}
void randomvalue(vector<vector<int>> &A, int row, int column)
{   cout<<"\n randomvalue entered;"<<row<<" "<<column<<" \n";
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 1000);
    for(int i=0;i<row;i++)
    for(int j=0;j<column;j++)
    {
        A[i][j]=(i+j)+distrib(gen);
    }
}

std::vector<std::vector<int>>  matrixadd(vector<vector<int>> &A, vector<vector<int>>&B)
{
    auto startadd=high_resolution_clock::now();
cout<<"\n the time started at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
    if(A.size()!=A[0].size())
    return {{}};
    else if(B.size()!=B[0].size())
    return {{}};
    if(A.size()!=B.size())
    return {{}};
    if(A[0].size()!=B[0].size())
    return {{}};
    vector<vector<int>> c(A[0].size(),vector<int>(B.size(),0));
  for(int i=0;i<A.size();i++)
  for(int j=0;j<B[0].size();j++)
  {
     c[i][j]=A[i][j]+B[i][j];
  }
  auto endadd=high_resolution_clock::now();
cout<<"\n the timetaken to finish matrix addition is: "<<duration<float,milli>(endadd-startadd).count()<<" ms\n";
cout<<" the result matrix is :\n";
printes(c);
  return c;
}

std::vector<std::vector<int>>  matrixsub(vector<vector<int>>&A, vector<vector<int>>&B)
{    auto startadd=high_resolution_clock::now();
cout<<"\n the time started at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
   if(A.size()!=A[0].size())
    return {{}};
    else if(B.size()!=B[0].size())
    return {{}};
    if(A.size()!=B.size())
    return {{}};
    if(A[0].size()!=B[0].size())
    return {{}};
    vector<vector<int>> c(A[0].size(),vector<int>(B.size(),0));
  for(int i=0;i<A.size();i++)
  for(int j=0;j<B[0].size();j++)
  {
     c[i][j]=A[i][j]-B[i][j];
  }
  auto endadd=high_resolution_clock::now();
cout<<"\n the timetaken to finish matrix subtraction is: "<<duration<float,milli>(endadd-startadd).count()<<" ms\n";
cout<<" the result matrix is :\n";
printes(c);
  return c;
}

std::vector<std::vector<int>>  matrixmul(vector<vector<int>>&A, vector<vector<int>>&B)
{
      auto startadd=high_resolution_clock::now();
cout<<"\n the time started at: "<<duration<float,milli>(startadd-start).count()<<" ms\n";
if(A[0].size()!=B.size())
return{{}};
   vector<vector<int>> c(A[0].size(),vector<int>(B.size(),0));
  for(int i=0;i<A.size();i++)
  for(int j=0;j<B[0].size();j++)
  {
     c[i][j]+=A[i][j]*B[j][i];
  }
  auto endadd=high_resolution_clock::now();
cout<<"\n the timetaken to finish matrix multiplication is: "<<duration<float,milli>(endadd-startadd).count()<<" ms\n";
cout<<" the result matrix is :\n";
printes(c);
  return c;
}

}