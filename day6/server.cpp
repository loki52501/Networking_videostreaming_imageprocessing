#undef unicode

#define DEFAULT_PORT "8080"
#define DEFAULT_BUFLEN 512

#define WIN32_LEAN_AND_MEAN

#include<windows.h>
#include<winsock2.h>
#include<ws2tcpip.h>
#include<iostream>
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



int main()
{
 if(!windowsocket_setup())
return -1;

addrinfo *ptr,*result,hints;
ZeroMemory(&hints,sizeof(hints));
hints.ai_socktype=SOCK_STREAM;
hints.ai_protocol=IPPROTO_TCP;
hints.ai_family=AF_INET;
hints.ai_flags=AI_PASSIVE;

int ires=getaddrinfo(NULL,DEFAULT_PORT,&hints,&result);
//address checker.
if(ires!=0)
{
cout<<"address not taken correctly or you did something baddd:"<<ires<<"\n";
WSACleanup();
return -1;
}

SOCKET Lsock=socket_create(*result);
if(Lsock==INVALID_SOCKET)
{
    cout<<"socket couldn't be created for whateverreason:"<<Lsock<<"\n";
    WSACleanup();
    return -1;
}
ires=bind(Lsock,result->ai_addr,result->ai_addrlen);
if(ires==SOCKET_ERROR)
{
    cout<<" something funny stopped me from binding, it's either port is used by something else or you're running this program."<<"\n";
    freeaddrinfo(result);
    WSACleanup();
    closesocket(Lsock);
    return -1;
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
while(true)
{
 Csock=accept(Lsock,NULL,NULL);
if(Csock==INVALID_SOCKET)
{
    cout<<" what the heck happened.. so far it was good.. "<<WSAGetLastError()<<"\n";
        WSACleanup();
    closesocket(Lsock);
    return -1;
}

char recvbuf[DEFAULT_BUFLEN];
int isres;
do{
 ires=recv(Csock,recvbuf,DEFAULT_BUFLEN,0);
if(ires>0)
{
    char SE[DEFAULT_BUFLEN]="look how i come";
            
    printf("bytes recieved:%d\n",ires);
    isres=send(Csock,SE,strlen(SE),0);
    if(isres==SOCKET_ERROR)
    {
        cout<<" failed to send:"<<WSAGetLastError<<"\n";
        
        closesocket(Lsock);
        closesocket(Csock);
        WSACleanup();
        return -1;
    }
          printf("bytes sent:%d\n",isres);
}
else if(ires==0)
{printf("\n the actual send was %s",recvbuf);
    cout<<"connection closed\n";

}
    else{
        printf("recv failed:%d\n",WSAGetLastError());
        closesocket(Csock);
        closesocket(Lsock);
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
    closesocket(Csock);
}

    
                closesocket(Lsock);

        WSACleanup();
        return -1;

}