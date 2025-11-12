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
