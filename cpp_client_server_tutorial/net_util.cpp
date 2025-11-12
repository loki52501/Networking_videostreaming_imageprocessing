#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAND_AND_MEAN
#endif
#define DEFAULT_PORT "27015"
#include<windows.h>
#include<winsock2.h>
#include<ws2tcpip.h>
#include<iphlpapi.h>
#include<stdio.h>


#pragma comment(lib,"Ws2_32.lib")

int main(){
    WSADATA wsad;
    int ires;
    struct addrinfo *result =NULL, *ptr=NULL,hints;
    ZeroMemory(&hints,sizeof(hints));
    hints.ai_family=AF_UNSPEC;
    hints.ai_socktype=SOCK_STREAM;
    hints.ai_protocol=IPPROTO_TCP;
    ires=getaddrinfo("localhost",DEFAULT_PORT,&hints,&result);
    if(ires!=0)
    {printf("getaddr failed:%d\n",ires);
    WSACleanup();
    return 1;
    }
ptr=result;
    SOCKET ConnectSocket=socket(ptr->ai_family,ptr->ai_socktype,ptr->ai_protocol);
    if(ConnectSocket==INVALID_SOCKET)
    {
        printf("ba ba blacksheep this is wrong : %ld\n",WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }
// Connect to server.
ires = connect( ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
if (ires == SOCKET_ERROR) {
    closesocket(ConnectSocket);
    ConnectSocket = INVALID_SOCKET;
}

// Should really try the next address returned by getaddrinfo
// if the connect call failed
// But for this simple example we just free the resources
// returned by getaddrinfo and print an error message

freeaddrinfo(result);

if (ConnectSocket == INVALID_SOCKET) {
    printf("Unable to connect to server!\n");
    WSACleanup();
    return 1;
    return 0;
}