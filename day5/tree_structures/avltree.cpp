#include<iostream>
#include<chrono>
using namespace std;
struct Node{
    int key;
    int height;
    Node* lc;
    Node* rc;
};

class avltree{
    Node* root;
    public:
    avltree(){root=new Node;}
    void rotateright(){
        
    }

    void rotateleft(){}
}