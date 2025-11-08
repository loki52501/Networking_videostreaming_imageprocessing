#undef unicode
#define WIN32_LEAN_AND_MEAN
#define DEFAULT_PORT "8000"
#define BUFF_SIZE 512

#include<iostream>
#include<windows.h>
#include<winsock2.h>
#include<WS2tcpip.h>
#include<iphlpapi.h>
#include<deque>
using namespace std;

template<typename T>
class client{
    deque<T> message;
    SOCKET Csock;
    addrinfo *result, hints;
    public:
    void startup()
    {
        WSADATA wsd;
        int rs=WSAStartup(MAKEWORD(2,2),&wsd);
        if(rs!=0)
        {
            cout<<" something wrong at startup;\n";
            return ;
        }
    }
    void Name(int family, int protocol, int sockettype)
    {
        ZeroMemory(&hints,sizeof(hints));
        hints.ai_family=family;
        hints.ai_protocol=protocol;
        hints.ai_socktype=sockettype;
        
    }
    void addressing()
    {
        int ires=getaddrinfo("127.0.0.1",DEFAULT_PORT,&hints,&result);
        if(ires!=0)
        {
            cout<<" the address has no memory or something is wrong with your declaration:\n";
            return;
        }

    }
    void create_socket()
    {
     Csock=socket(result->ai_family,result->ai_socktype,result->ai_protocol);
     if(Csock==INVALID_SOCKET)
     {
        cout<<"something went wrong when creating connectionsocket:\n"<<WSAGetLastError();
return;
     }   

    }
    void connection()
    {
        int ires=connect(Csock,result->ai_addr,result->ai_addrlen);
        if(ires==SOCKET_ERROR)
        {
            cout<<" socket error during connection \n";
           closesocket(Csock);
            return;        }
    }
    void send_recieve()
    {
        
        int ires;
        char recvs[BUFF_SIZE],sends[BUFF_SIZE]="i'm good and i'm bad , let s meet server.\n";
        ires=send(Csock,sends,(int)strlen(sends),0);
       
        if(ires==SOCKET_ERROR)
        {
            cout<<" sending made some error;"<<WSAGetLastError()<<"\n";
shutdowns();
        }
         if(shutdown(Csock,SD_SEND)==SOCKET_ERROR)
        {
            cout<<" everything went haywire\n";
            return;
        }
        cout<<" sent made: now \n";
        do{
            ires= recv(Csock,recvs,BUFF_SIZE,0);
            if(ires>0)    
            cout<<"the receiver said:"<<recvs<<"\n";
            else if(ires==0)
            {
                cout<<" finished receiving: good, connection closed;\n";
shutdowns();
            }
            else{
                cout<<" something went wrong;"<<WSAGetLastError()<<"\n";
shutdowns();            }

        }while(ires>0);

    }
    void shutdowns()
    {
        closesocket(Csock);
        WSACleanup();

        return;
    }
    void client_start()
{

    Name(AF_INET,IPPROTO_TCP,SOCK_STREAM);
    addressing();
    create_socket();
    connection();
    send_recieve();
    shutdowns();
    return ;
}

};


