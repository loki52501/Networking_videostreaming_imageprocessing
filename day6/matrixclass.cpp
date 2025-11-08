#include<iostream>
#include<vector>
#include<random>

using namespace std;

typedef vector<vector<int>> tensor;
class Matrix{
    tensor A,B,C;
    public:
  void randomvalue(tensor &A,int row, int column)
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
Matrix()
{
  randomvalue(A,4,4);
  randomvalue(B,4,4);
}
void printes(tensor&A)
{if(A.size()!=0)
    for(int i=0;i<A.size();i++)
    {for(int j=0;j<A[0].size();j++)
    {
        cout<<"|"<<A[i][j]<<" ";
    }
    cout<<"|\n";
}
}
tensor matrixadd(tensor &A, tensor&B)
{
 
    if(A.size()!=A[0].size())
    return {{}};
    else if(B.size()!=B[0].size())
    return {{}};
    if(A.size()!=B.size())
    return {{}};
    if(A[0].size()!=B[0].size())
    return {{}};
    tensor c(A[0].size(),vector<int>(B.size(),0));
  for(int i=0;i<A.size();i++)
  for(int j=0;j<B[0].size();j++)
  {
     c[i][j]=A[i][j]+B[i][j];
  }
cout<<" the result matrix is :\n";
printes(c);
  return c;
}

};