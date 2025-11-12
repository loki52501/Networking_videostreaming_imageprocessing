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
  explicit socket_guard(socket_handle h = invalid_socket) noexcept : handle_(h) {}
  ~socket_guard() { reset(); }

  socket_guard(socket_guard&& other) noexcept : handle_(other.release()) {}
  socket_guard& operator=(socket_guard&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = other.release();
    }
    return *this;
  }

  socket_guard(const socket_guard&) = delete;
  socket_guard& operator=(const socket_guard&) = delete;

  socket_handle get() const noexcept { return handle_; }
  socket_handle release() noexcept {
    socket_handle old = handle_;
    handle_ = invalid_socket;
    return old;
  }
  void reset(socket_handle h = invalid_socket) noexcept {
    if (handle_ != invalid_socket) {
      close_socket(handle_);
    }
    handle_ = h;
  }

private:
  socket_handle handle_;
};

socket_guard make_tcp_socket();
void bind_and_listen(socket_handle server, std::uint16_t port, int backlog = 1);
socket_guard accept_client(socket_handle server);
void connect_to(socket_handle client, const std::string& host, std::uint16_t port);
void send_line(socket_handle sock, const std::string& line);
std::string recv_line(socket_handle sock);
