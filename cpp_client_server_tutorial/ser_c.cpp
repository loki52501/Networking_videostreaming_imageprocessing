#undef UNICODE

#define WIN32_LEAN_AND_MEAN
#define DEFAULT_PORT "23456"
#define DEFAULT_BUFLEN 512

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include<iphlpapi.h>
#include <stdio.h>
// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")

int main(){
WSADATA wsd;
int iRes=WSAStartup(MAKEWORD(2,2),&wsd);

if(iRes != 0) {
    printf("WSAStartup failed: %d\n", iRes);
    return 1;
}

addrinfo *ptr=NULL,*result=NULL,hints;

ZeroMemory(&hints,sizeof(hints));
hints.ai_family=AF_UNSPEC;
hints.ai_socktype=SOCK_STREAM;
hints.ai_protocol=IPPROTO_TCP;

iRes=getaddrinfo("127.0.0.1",DEFAULT_PORT,&hints,&result);
SOCKET Csock=INVALID_SOCKET;

ptr=result;
Csock=socket(ptr->ai_family,ptr->ai_socktype,ptr->ai_protocol);
if(Csock==INVALID_SOCKET)
{
    printf("something is going boogy %d\n",WSAGetLastError());
    freeaddrinfo(result);
    WSACleanup();
    return 1;
}
printf("client connected lal lalal al: %d , this is result: %d",Csock,result->ai_next);

iRes=connect(Csock,ptr->ai_addr,(int)ptr->ai_addrlen);
if(iRes==SOCKET_ERROR)
{printf("\nwell. concetion went far way\n");
    closesocket(Csock);
    Csock=INVALID_SOCKET;
}
freeaddrinfo(result);

if(Csock==INVALID_SOCKET)
{
printf("you're dead.. go awy , unable to connect server\n");
WSACleanup();
return 1;
}

const char *sendbuf=" this is haloweeey skd";
char recvbuf[DEFAULT_BUFLEN];

iRes=send(Csock,sendbuf,(int)strlen(sendbuf),0);
if(iRes==SOCKET_ERROR)
{
    printf("send failed:%d\n",WSAGetLastError());
    closesocket(Csock);
    WSACleanup();
    return 1;
}
printf("Bytes send yay: %ld\n",iRes);
iRes=shutdown(Csock,SD_SEND);
if(iRes==SOCKET_ERROR)
{
    printf("shutdown failed: %d\n", WSAGetLastError);
    closesocket(Csock);
    WSACleanup();
    return 1;
}
do{
    iRes=recv(Csock,recvbuf,DEFAULT_BUFLEN,0);
      if (iRes > 0)
        printf("Bytes received: %d and the content is: %s\n", iRes,recvbuf);
    else if (iRes == 0)
        printf("Connection closed\n");
    else
        printf("recv failed: %d\n", WSAGetLastError());
} while (iRes > 0);
closesocket(Csock);
WSACleanup();
return 0;
}