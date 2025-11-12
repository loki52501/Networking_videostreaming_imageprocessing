# C++ TCP Client-Server Starter

## What You Will Build
By the end of these exercises you will have two small programs:
- a `server` that listens on `127.0.0.1:3333`, accepts a single TCP client, and echoes each line it receives;
- a `client` that connects to the server, sends lines from standard input, and prints the echo replies.

You can extend the same pattern to richer application protocols after you understand the concepts below.

## Prerequisites
- Modern C++ (C++17 or newer) and comfort with RAII.
- A compiler toolchain (`g++`, `clang++`, or MSVC `cl`) and CMake 3.15 or newer.
- Basic command-line knowledge.
- On Windows: Windows SDK or Visual Studio Build Tools so that Winsock2 headers and libraries are available.

## Conceptual Foundations
### 1. Transport vs Application Protocols
TCP gives you a reliable byte stream between two sockets. It does not define message boundaries. Your application protocol must decide how messages are delimited (newline, fixed-length header, length-prefixed binary, etc.). Keep this in mind while designing both client and server.

### 2. Socket Addressing
- Address family: `AF_INET` for IPv4, `AF_INET6` for IPv6.
- Socket type: `SOCK_STREAM` for TCP.
- Protocol: usually `IPPROTO_TCP`.
Windows requires one call to `WSAStartup` before the first socket is created. POSIX systems do not.

### 3. Server Lifecycle
1. `socket()` to obtain a listening socket.
2. `bind()` to attach it to an IP address and port.
3. `listen()` to transition it into a passive socket.
4. `accept()` to obtain a new connected socket per client.
5. Loop on `recv()` and `send()` until the client closes the connection.

### 4. Client Lifecycle
1. `socket()` to create a TCP socket.
2. `connect()` to the server's address.
3. Exchange data with `send()` and `recv()`.
4. Close the socket cleanly to release resources.

### 5. Blocking I/O and Timeouts
The simplest model uses blocking sockets: each call waits for completion. Configure reasonable timeouts with `setsockopt` (`SO_RCVTIMEO`, `SO_SNDTIMEO`) so you can recover if the peer disappears. Later you can explore non-blocking sockets with `select`, `poll`, `epoll`, or IOCP.

### 6. Byte Ordering
Network byte order is big-endian. Use helpers `htons`, `htonl`, `ntohs`, `ntohl` whenever you move integers between host memory and the network to keep the code portable.

### 7. Resource Management
Always release sockets. A simple RAII wrapper around the native handle ensures `closesocket` (Windows) or `close` (POSIX) runs even when exceptions happen.

## Hands-On Walkthrough
Work inside the `cpp_client_server_tutorial/` folder.

### 1. Create the Project Skeleton
```
cpp_client_server_tutorial/
  CMakeLists.txt
  include/
    net_util.hpp
  src/
    net_util.cpp
    server.cpp
    client.cpp
  README.md
```

Create the directories:
```powershell
mkdir include, src
```

### 2. Write `include/net_util.hpp`
Declare helpers that hide platform differences and give you RAII ownership of sockets.
```cpp
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socket_handle = SOCKET;
  constexpr socket_handle invalid_socket = INVALID_SOCKET;
  inline void close_socket(socket_handle s) { closesocket(s); }
#else
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <sys/socket.h>
  #include <unistd.h>
  using socket_handle = int;
  constexpr socket_handle invalid_socket = -1;
  inline void close_socket(socket_handle s) { ::close(s); }
#endif

struct winsock_session {
  winsock_session();
  ~winsock_session();
};

struct socket_guard {
  explicit socket_guard(socket_handle h = invalid_socket) noexcept : handle(h) {}
  ~socket_guard() { reset(); }

  socket_guard(socket_guard&& other) noexcept : handle(other.release()) {}
  socket_guard& operator=(socket_guard&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.release();
    }
    return *this;
  }

  socket_guard(const socket_guard&) = delete;
  socket_guard& operator=(const socket_guard&) = delete;

  socket_handle get() const noexcept { return handle; }
  socket_handle release() noexcept {
    socket_handle old = handle;
    handle = invalid_socket;
    return old;
  }
  void reset(socket_handle h = invalid_socket) noexcept {
    if (handle != invalid_socket) {
      close_socket(handle);
    }
    handle = h;
  }

private:
  socket_handle handle;
};

socket_guard make_tcp_socket();
void bind_and_listen(socket_handle server, std::uint16_t port, int backlog = 1);
socket_guard accept_client(socket_handle server);
void connect_to(socket_handle client, const std::string& host, std::uint16_t port);
void send_line(socket_handle sock, const std::string& line);
std::string recv_line(socket_handle sock);
```

### 3. Implement `src/net_util.cpp`
```cpp
#include "net_util.hpp"

winsock_session::winsock_session() {
#ifdef _WIN32
  WSADATA data;
  int err = WSAStartup(MAKEWORD(2, 2), &data);
  if (err != 0) {
    throw std::runtime_error("WSAStartup failed");
  }
#endif
}

winsock_session::~winsock_session() {
#ifdef _WIN32
  WSACleanup();
#endif
}

socket_guard make_tcp_socket() {
  socket_handle s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == invalid_socket) {
    throw std::runtime_error("socket() failed");
  }
  return socket_guard{s};
}

void bind_and_listen(socket_handle server, std::uint16_t port, int backlog) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);

  int opt = 1;
  ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == -1) {
    throw std::runtime_error("bind() failed");
  }
  if (::listen(server, backlog) == -1) {
    throw std::runtime_error("listen() failed");
  }
}

socket_guard accept_client(socket_handle server) {
  socket_handle client = ::accept(server, nullptr, nullptr);
  if (client == invalid_socket) {
    throw std::runtime_error("accept() failed");
  }
  return socket_guard{client};
}

void connect_to(socket_handle client, const std::string& host, std::uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  addrinfo* head = nullptr;
  int err = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &head);
  if (err != 0) {
#ifdef _WIN32
    throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerrorA(err)));
#else
    throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerror(err)));
#endif
  }

  for (addrinfo* p = head; p; p = p->ai_next) {
    if (::connect(client, p->ai_addr, static_cast<int>(p->ai_addrlen)) == 0) {
      ::freeaddrinfo(head);
      return;
    }
  }

  ::freeaddrinfo(head);
  throw std::runtime_error("connect() failed");
}

void send_line(socket_handle sock, const std::string& line) {
  std::string payload = line;
  payload.push_back('\n');

  std::size_t total = 0;
  while (total < payload.size()) {
    int sent = ::send(sock, payload.data() + total,
                      static_cast<int>(payload.size() - total), 0);
    if (sent <= 0) {
      throw std::runtime_error("send() failed");
    }
    total += static_cast<std::size_t>(sent);
  }
}

std::string recv_line(socket_handle sock) {
  std::string buffer;
  char ch = '\0';

  while (true) {
    int received = ::recv(sock, &ch, 1, 0);
    if (received == 0) {
      throw std::runtime_error("connection closed by peer");
    }
    if (received < 0) {
      throw std::runtime_error("recv() failed");
    }
    if (ch == '\n') {
      break;
    }
    buffer.push_back(ch);
  }

  return buffer;
}
```

### 4. Write `src/server.cpp`
```cpp
#include "net_util.hpp"
#include <iostream>

int main() {
  try {
    winsock_session winsock_guard;
    socket_guard server = make_tcp_socket();

    bind_and_listen(server.get(), 3333);
    std::cout << "Server listening on 127.0.0.1:3333\n";

    socket_guard client = accept_client(server.get());
    std::cout << "Client connected\n";

    while (true) {
      std::string line = recv_line(client.get());
      std::cout << "recv: " << line << '\n';
      send_line(client.get(), line);
    }
  } catch (const std::exception& ex) {
    std::cerr << "Fatal error: " << ex.what() << '\n';
    return 1;
  }
}
```

### 5. Write `src/client.cpp`
```cpp
#include "net_util.hpp"
#include <iostream>

int main() {
  try {
    winsock_session winsock_guard;
    socket_guard client = make_tcp_socket();

    connect_to(client.get(), "127.0.0.1", 3333);
    std::cout << "Connected to server. Type messages, press Ctrl+Z (Windows) or Ctrl+D (POSIX) to exit.\n";

    std::string line;
    while (std::getline(std::cin, line)) {
      send_line(client.get(), line);
      std::cout << "echo: " << recv_line(client.get()) << '\n';
    }
  } catch (const std::exception& ex) {
    std::cerr << "Fatal error: " << ex.what() << '\n';
    return 1;
  }
}
```

### 6. Add `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.15)
project(cpp_client_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(net_util src/net_util.cpp)

add_executable(server src/server.cpp)
target_link_libraries(server PRIVATE net_util)

add_executable(client src/client.cpp)
target_link_libraries(client PRIVATE net_util)

if (WIN32)
  target_link_libraries(server PRIVATE ws2_32)
  target_link_libraries(client PRIVATE ws2_32)
endif()
```

### 7. Configure and Build
```powershell
cmake -S . -B build
cmake --build build --config Release
```
On single-config generators (MinGW Makefiles, Ninja) the binaries live directly under `build/`. On Visual Studio, use `build\Release\`.

### 8. Run the Programs
Open two terminals.

Server:
```powershell
.\build\Release\server.exe
```

Client:
```powershell
.\build\Release\client.exe
```

Type into the client terminal and confirm the server echoes the lines.

### 9. Experiments to Deepen Understanding
- Add a two-byte length prefix to every message and update `send_line` / `recv_line` to use it instead of newline framing.
- Handle multiple clients by spawning a `std::thread` per accepted connection and protecting shared state with mutexes.
- Switch to non-blocking sockets and use `select` to multiplex I/O.
- Add graceful shutdown (`shutdown()` before closing the socket) so the peer notices the end-of-stream cleanly.
- Log error codes (`WSAGetLastError()` or `errno`) before throwing to speed up debugging.

### 10. Debugging Checklist
- If `connect()` fails, ensure the server is running and listening on the expected port (use `netstat -an`).
- Use `telnet 127.0.0.1 3333` or `nc 127.0.0.1 3333` to sanity-check the server.
- Confirm you called `winsock_session` before creating sockets on Windows.
- Verify firewalls allow `127.0.0.1:3333`.

## Next Steps
Once the echo example works, explore:
- Transport Layer Security with OpenSSL or the standalone `asio` library.
- Structured application protocols (Protocol Buffers, JSON, FlatBuffers).
- High-performance event loops (Boost.Asio, `std::experimental::net`, libuv).
- UDP-based protocols and reliability strategies.
