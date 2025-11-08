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
void addressing()
{
    int ires=getaddrinfo(NULL,DEFAULT_PORT,&hints,&result);
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
        cout<<" listening sock is invalid";
        return;
    }
    return;
}
void client_socket_creation()
{
    Csock=accept(Lsock,NULL,NULL);
    if(Csock==INVALID_SOCKET)
    {
           cout<<"socket invalid";
           return;
    }
    closesocket(Lsock);
    freeaddrinfo(result);
    return;
}
void binding()
{
    int ires=bind(Lsock,result->ai_addr,result->ai_addrlen);
    if(ires==SOCKET_ERROR)
    {cout<<" socket binding went wrong";
        WSACleanup();
        closesocket(Lsock);
    return;
    }
}

void listening()
{
    if(listen(Lsock,SOMAXCONN)==SOCKET_ERROR){
    cout<<" socket listening made some mistakes , go and look around";
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
void shutdowns()
{
    shutdown(Csock,SD_SEND);
closesocket(Csock);
WSACleanup();

return;
}
};

int main()
{
    server<string> se;
    se.Wsastartup();
    se.Name(AF_INET,IPPROTO_TCP,SOCK_STREAM);
    se.addressing();
    se.socket_creation();
    se.binding();
    se.listening();
    se.client_socket_creation();
    se.recieving_sending();
    se.shutdowns();
    return -1;
}