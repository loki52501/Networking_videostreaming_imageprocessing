#include "include/net_util.hpp"

#ifdef _WIN32
  #pragma comment(lib, "ws2_32.lib")
#endif

winsock_session::winsock_session() {
#ifdef _WIN32
  WSADATA data;
  const int err = WSAStartup(MAKEWORD(2, 2), &data);
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

  int reuse = 1;
  ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&reuse), sizeof(reuse));

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
  const int err = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &head);
  if (err != 0) {
#ifdef _WIN32
    throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerrorA(err)));
#else
    throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerror(err)));
#endif
  }

  for (addrinfo* node = head; node; node = node->ai_next) {
    if (::connect(client, node->ai_addr, static_cast<int>(node->ai_addrlen)) == 0) {
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
    const int sent = ::send(sock, payload.data() + total,
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
    const int received = ::recv(sock, &ch, 1, 0);
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
