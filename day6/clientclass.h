#pragma once
#undef unicode
#define WIN32_LEAN_AND_MEAN
#define DEFAULT_PORT "8000"
#define BUFF_SIZE 512

#include <deque>
#include <iostream>
#include <windows.h>
#include <winsock2.h>
#include <WS2tcpip.h>
#include <iphlpapi.h>
#include<string>
template<typename T>
class client {
    std::deque<T> message;
    SOCKET Csock = INVALID_SOCKET;
    addrinfo* result = nullptr;
    addrinfo hints{};

public:
    void startup() {
        WSADATA wsd;
        if (WSAStartup(MAKEWORD(2, 2), &wsd) != 0) {
            std::cout << " something wrong at startup;\n";
        }
    }

    void Name(int family, int protocol, int sockettype) {
        ZeroMemory(&hints, sizeof(hints));
        hints.ai_family = family;
        hints.ai_protocol = protocol;
        hints.ai_socktype = sockettype;
    }

    void addressing(std::string port=DEFAULT_PORT) {
        if (getaddrinfo("127.0.0.1",port.c_str() , &hints, &result) != 0) {
            std::cout << " the address has no memory or something is wrong with your declaration:\n";
        }
    }

    void create_socket() {
        Csock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (Csock == INVALID_SOCKET) {
            std::cout << "something went wrong when creating connectionsocket:\n" << WSAGetLastError();
        }
    }

    void connection() {
        if (connect(Csock, result->ai_addr, result->ai_addrlen) == SOCKET_ERROR) {
            std::cout << " socket error during connection \n";
            closesocket(Csock);
        }
    }

    void send_recieve() {
        int ires;
        char recvs[BUFF_SIZE];
        char sends[BUFF_SIZE] = "i'm good and i'm bad , let s meet server.\n";

        ires = send(Csock, sends, static_cast<int>(strlen(sends)), 0);
        if (ires == SOCKET_ERROR) {
            std::cout << " sending made some error;" << WSAGetLastError() << "\n";
            shutdowns();
            return;
        }

        if (shutdown(Csock, SD_SEND) == SOCKET_ERROR) {
            std::cout << " everything went haywire\n";
            return;
        }

        std::cout << " sent made: now \n";
        do {
            ires = recv(Csock, recvs, BUFF_SIZE, 0);
            if (ires > 0) {
                recvs[ires] = '\0';
                std::cout << "the receiver said:" << recvs << "\n";
            } else if (ires == 0) {
                std::cout << " finished receiving: good, connection closed;\n";
                shutdowns();
            } else {
                std::cout << " something went wrong;" << WSAGetLastError() << "\n";
                shutdowns();
            }
        } while (ires > 0);
    }

    void shutdowns() {
        closesocket(Csock);
     
    }

    void client_start(std::string port) {

        Name(AF_INET, IPPROTO_TCP, SOCK_STREAM);
        addressing(port);
        create_socket();
        connection();
        send_recieve();
           WSACleanup();

    }
    ~client()
    {
        shutdowns();
    }
};
