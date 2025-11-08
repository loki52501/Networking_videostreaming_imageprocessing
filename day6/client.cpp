#undef unicode

#define DEFAULT_PORT "8080"
#define BUFF_LEN 512
#define WIN32_LEAN_AND_MEAN

#include<iostream>
#include<windows.h>
#include<WinSock2.h>
#include<WS2tcpip.h>
#include<iphlpapi.h>
using namespace std;

bool windowsocket_setup()
{
    WSADATA wsd;
    int ires=WSAStartup(MAKEWORD(2,2),&wsd);
    if(ires!=0)
    {
        cout<<"everything went to hell because you haven't linked the library ws2_32";
        return false;
    }
return true;
}

SOCKET socket_create(addrinfo dats)
{
 SOCKET a=socket(dats.ai_family,dats.ai_socktype,dats.ai_protocol);
 return a;
}

void create_addr(addrinfo &hints)
{
    ZeroMemory(&hints,sizeof(hints));
hints.ai_family=AF_INET;
hints.ai_protocol=IPPROTO_TCP;
hints.ai_socktype=SOCK_STREAM;
}

int main()
{
    if(!windowsocket_setup())
    return -1;
    addrinfo *ptr,*result,hints;
    create_addr(hints);
    int ires=getaddrinfo("127.0.0.1",DEFAULT_PORT,&hints,&result);
    cout<<hints.ai_family;
    if(ires!=0)
    {
        cout<<"addressinfo entering went wrong somewhere:"<<WSAGetLastError();
      return -1;
    }
    SOCKET Csock=socket_create(*result);
    if(Csock==INVALID_SOCKET)
    {
          cout<<"client socket what what:"<<WSAGetLastError();
          freeaddrinfo(result);
      WSACleanup();
      return -1;
    }
    ires=connect(Csock,result->ai_addr,result->ai_addrlen);
    if(ires==SOCKET_ERROR)
    {
        cout<<"csocket went haywire when connecting to the server:"<<WSAGetLastError();
        closesocket(Csock);
      WSACleanup();
      return -1;
    }
    freeaddrinfo(result);

    const char* sendbuf="long super duper";
    int ir=send(Csock,sendbuf,(int)strlen(sendbuf),0);
    if(ir==SOCKET_ERROR)
    {
        cout<<"send failed oops"<<WSAGetLastError();
         closesocket(Csock);
      WSACleanup();
        return -1;
    }
printf("Bytes send yay: %ld\n",ires);
ires=shutdown(Csock,SD_SEND);
if(ires==SOCKET_ERROR)
{
    cout<<"shutdown somethwer sewrong\n";
      closesocket(Csock);
      WSACleanup();
        return -1;
}

    do{
        char recvbuf[BUFF_LEN];
    ires=recv(Csock,recvbuf,BUFF_LEN,0);
      if (ires > 0)
        printf("Bytes received: %d and the content is: %s\n", ires,recvbuf);
    else if (ires == 0)
        printf("Connection closed\n");
    else
        printf("recv failed: %d\n", WSAGetLastError());
} while (ires > 0);

closesocket(Csock);
WSACleanup();
return -1;
}