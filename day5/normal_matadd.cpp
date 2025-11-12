#include <future>
#include <iostream>
#include <thread>
#include<random>
#include<chrono>
using namespace std;
using namespace chrono;
using Matrix = std::vector<std::vector<int>>;

void matrixadd(Matrix A,Matrix B,Matrix &C,int row, int column)
{
  for(int i=0;i<row;i++)
  for(int j=0;j<column;j++)
  C[i][j]=A[i][j]+B[i][j];
}
void printes(Matrix&A)
{if(A.size()!=0)
    for(int i=0;i<A.size();i++)
    {for(int j=0;j<A[0].size();j++)
    {
        cout<<"|"<<A[i][j]<<" ";
    }
    cout<<"|\n";
}
}
void randomvalue(Matrix &A, int row, int column)
{   cout<<"\n randomvalue entered;"<<row<<" "<<column<<" \n";
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 20);
    for(int i=0;i<row;i++)
    for(int j=0;j<column;j++)
    {
        A[i][j]=(i+j)+distrib(gen);
    }
}
int main()
{
    auto start =high_resolution_clock::now();
 Matrix A{2,vector<int>(2,0)},B{2,vector<int>(2,0)},C{2,vector<int>(2,0)};
 randomvalue(A,2,2);
 randomvalue(B,2,2);
 cout<<"\n this is matrixA:\n";
 printes(A);
  cout<<"\n this is matrixB:\n";
 printes(B);
 matrixadd(A,B,C,2,2);
  cout<<" \n the result matrix is :\n";
 printes(C);
 auto end=high_resolution_clock::now();
 cout<<" \n total time taken: "<<duration<float,milli>(end-start).count()<<"ms\n";
 return 0;
}