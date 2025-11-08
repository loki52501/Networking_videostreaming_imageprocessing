#include<iostream>
#include<vector>
#include<random>

using namespace std;

typedef vector<vector<int>> tensor;
class Matrix{
    
    public:
    tensor A,B,C;
  void randomvalue(tensor &D,int row, int column)
{   D.assign(row, std::vector<int>(column));   // allocate storage
    // then fill m[i][j]
     cout<<"\n randomvalue entered;"<<row<<" "<<column<<" \n";
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 100);
    for(int i=0;i<row;i++)
    for(int j=0;j<column;j++)
    {
        D[i][j]=(i+j)+distrib(gen);
    }
}
Matrix()
{
  randomvalue(A,4,4);
  randomvalue(B,4,4);
   C.assign(4, std::vector<int>(4)); 
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
int matrixadd(tensor &A, tensor&B, int row, int column)
{
 
C[row][column]=A[row][column]+B[row][column];

cout<<" i added this"<<A[row][column]<<" + "<<B[row][column]<<" = "<<C[row][column]<<"\n";
return C[row][column];
}

};