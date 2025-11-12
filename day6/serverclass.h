#pragma once
#undef unicode
#define BUFF_SIZE 512
#define DEFAULT_PORT "8000"
#define WIN32_LEAN_AND_MEAN

#include<iostream>
#include<deque>
#include<windows.h>
#include<winsock2.h>
#include<WS2tcpip.h>
#include<iphlpapi.h>
using namespace std;

template<typename T>
class server{
deque<T> messages;
SOCKET Csock=INVALID_SOCKET, Lsock=INVALID_SOCKET;
addrinfo *result , hints;
public:

server(string port)
{
    Name(AF_INET,IPPROTO_TCP,SOCK_STREAM);
    addressing(port);
    socket_creation();
  binding();
}

void Wsastartup()
{WSADATA WSD;
    int ires=WSAStartup(MAKEWORD(2,2),&WSD);
 if(ires!=0)
 {
    cout<< "WSA startup error look technician;";
    return;
 }
 return ;
}
void Name(int family, int protocol,int sockettype )
{ZeroMemory(&hints,sizeof(hints));

    hints.ai_family=family;
    hints.ai_protocol=protocol;
    hints.ai_socktype=sockettype;
}
void addressing(string port=DEFAULT_PORT)
{
    int ires=getaddrinfo(NULL,port.c_str(),&hints,&result);
    if(ires!=0)
    {
        cout<<" invalid address or couldn't get addrinfo"<<"\n";
        return ;
    }
}
void socket_creation()
{

    Lsock=socket(hints.ai_family,hints.ai_socktype,hints.ai_protocol);
    if(Lsock==INVALID_SOCKET)
    {
        cout<<" listening sock is invalid"<<WSAGetLastError();
        return;
    }
           BOOL reuse = TRUE;
    if(setsockopt(Lsock,SOL_SOCKET,SO_REUSEADDR,(char*)(&reuse),sizeof(reuse))==SOCKET_ERROR){
        cout<<"setsockopt(so_reuseaddr) failed: "<<WSAGetLastError();
        closesocket(Lsock);
        Lsock=INVALID_SOCKET;
    }
    return;
}
SOCKET client_socket_creation()
{
    Csock=accept(Lsock,NULL,NULL);
    if(Csock==INVALID_SOCKET)
    {
           cout<<"socket invalid"<<WSAGetLastError();
           return Csock;
    }

    freeaddrinfo(result);
    result=nullptr;
return Csock;
}
void binding()
{
    int ires=bind(Lsock,result->ai_addr,result->ai_addrlen);
    if(ires==SOCKET_ERROR)
    {cout<<" socket binding went wrong"<<WSAGetLastError();
        WSACleanup();
        closesocket(Lsock);
    return;
    }
}

void listening()
{
    if(Lsock!=INVALID_SOCKET)
    if(listen(Lsock,SOMAXCONN)==SOCKET_ERROR){
    cout<<" socket listening made some mistakes , go and look around"<<WSAGetLastError();
     WSACleanup();
        closesocket(Lsock);
    return ;
    }
 

}
void add_messages()
{
    T message;
    cout<<" enter your message to the client, now that you're connected;";
    cin>>message;
    messages.push_back(message);
}
void recieving_sending()
{
    char recvbuf[BUFF_SIZE],sendbuf[BUFF_SIZE]="something good;";
    int isr,is;
do{
   isr =recv(Csock,recvbuf,BUFF_SIZE,0);
   if(isr>0)
   {
    is=send(Csock,sendbuf,(int)strlen(sendbuf),0);
    if(is==SOCKET_ERROR)
    {cout<<"made wrong;"<<WSAGetLastError()<<"\n";
        return;
    }
    cout<<" client:"<<recvbuf<<"\n";
}

    else if(isr==0)
    {
        cout<<" connection closed:"<<recvbuf<<" this is what i received";
        return;
    }
    else
    { cout<<" something went wrong when recieving.. go figure out."<<WSAGetLastError()<<"\n";
}      
   }while(isr>0);

}
void handle_client(SOCKET Csock)
{
    char recvbuf[BUFF_SIZE],sendbuf[BUFF_SIZE]="something good;";
    int isr,is;
do{
   isr =recv(Csock,recvbuf,BUFF_SIZE,0);
   if(isr>0)
   {
    is=send(Csock,sendbuf,(int)strlen(sendbuf),0);
    if(is==SOCKET_ERROR)
    {cout<<"made wrong;"<<WSAGetLastError()<<"\n";
        return;
    }
    cout<<" client:"<<recvbuf<<"\n";
}

    else if(isr==0)
    {
        cout<<" connection closed:"<<recvbuf<<" this is what i received";
        return;
    }
    else
    { cout<<" something went wrong when recieving.. go figure out."<<WSAGetLastError()<<"\n";
}      
   }while(isr>0);

}
void shutdowns()
{
    shutdown(Csock,SD_SEND);
        closesocket(Lsock);
closesocket(Csock);


return;
}
void server_start(string port)
{

    Name(AF_INET,IPPROTO_TCP,SOCK_STREAM);
    addressing(port);
    socket_creation();
    binding();
    listening();
    client_socket_creation();
    recieving_sending();
    WSACleanup();
    return;
}

};

