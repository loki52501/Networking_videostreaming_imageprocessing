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

#include <iostream>
#include<string>
#include <iomanip>

void dump_addrinfo(const addrinfo* head) {
  std::cout << "addrinfo list:\n";
  for (auto node = head; node; node = node->ai_next) {
    std::cout << "  family="
              << (node->ai_family == AF_INET  ? "AF_INET" :
                  node->ai_family == AF_INET6 ? "AF_INET6" : std::to_string(node->ai_family))
              << " socktype="
              << (node->ai_socktype == SOCK_STREAM ? "SOCK_STREAM" :
                  node->ai_socktype == SOCK_DGRAM  ? "SOCK_DGRAM" :
                  std::to_string(node->ai_socktype))
              << " protocol=" << node->ai_protocol << '\n';

    char host[NI_MAXHOST] = {};
    if (getnameinfo(node->ai_addr, static_cast<DWORD>(node->ai_addrlen),
                    host, sizeof(host), nullptr, 0, NI_NUMERICHOST) == 0) {
      std::cout << "    addr=" << host << '\n';
    }
  }
}
int main()
{
    WSADATA WS;
    int ires;
    ires=WSAStartup(MAKEWORD(2,2),&WS);
    if(ires!=0)
    {
        printf(" the lib went boogy amd i stopped because of it. %d",ires);
        return 1;
    }
    addrinfo *ptr,*result,hints;
    ZeroMemory(&hints,sizeof(hints));
    hints.ai_family=AF_INET;
    hints.ai_socktype=SOCK_STREAM;
    hints.ai_protocol=IPPROTO_TCP;
    hints.ai_flags=AI_PASSIVE;
    ires=getaddrinfo(NULL,DEFAULT_PORT,&hints,&result);
 dump_addrinfo(result);
    if(ires!=0)
    {
        printf("address error, look:%d\n",ires);
        WSACleanup();
        return 1;
    }
    SOCKET Lsock=INVALID_SOCKET;
    Lsock=socket(result->ai_family,result->ai_socktype,result->ai_protocol);
    if(Lsock==INVALID_SOCKET)
    {
        printf("creation of socket went haywire please please:%d",Lsock);
        freeaddrinfo(result);
        WSACleanup();
        return -1;
    }
    ires=bind(Lsock,result->ai_addr,(int)result->ai_addrlen);
    if(ires==SOCKET_ERROR)
    {
        printf("binding went far away,please check the error : %d\n", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(Lsock);
        WSACleanup();
        return 1;
    }
    freeaddrinfo(result);
if(listen(Lsock,SOMAXCONN)==SOCKET_ERROR)
{
    printf("listen failed with erro:%ld\n",WSAGetLastError());
    closesocket(Lsock);
    WSACleanup();
    return 1;
}
SOCKET Csock=INVALID_SOCKET;
Csock=accept(Lsock,NULL,NULL);
if(Csock==INVALID_SOCKET)
{
    printf("accept failed:%d\n",WSAGetLastError());
    closesocket(Lsock);
    WSACleanup();
    return 1;
}
char recvbuf[DEFAULT_BUFLEN];
int isres;
do{
    ires=recv(Csock,recvbuf,DEFAULT_BUFLEN,0);
    if(ires>0){
        char recvs[DEFAULT_BUFLEN]="look how far i have come" ;

        printf("bytes recieved:%d\n",ires);
        isres=send(Csock,recvs,ires,0);
        if(isres==SOCKET_ERROR)
        {
            printf("send failed:%d\n",WSAGetLastError);
            closesocket(Csock);
            WSACleanup();
            return 1;
        }
        printf("bytes sent:%d\n",isres);

    }else if(ires==0)
    {  printf("\n the actual send was %s",recvbuf);
        printf("connection closing\n");
    }
    else{
        printf("recv failed:%d\n",WSAGetLastError());
        closesocket(Csock);
        WSACleanup();
        return 1;
    }
}while(ires>0);
ires=shutdown(Csock,SD_SEND);
if(ires==SOCKET_ERROR)
{
    printf("shutdown failed:%d\n",WSAGetLastError());
    closesocket(Csock);
    WSACleanup();
    return 1;
}
closesocket(Lsock);
closesocket(Csock);
WSACleanup();
return 1;
}