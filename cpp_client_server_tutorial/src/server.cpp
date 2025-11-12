#include "net_util.hpp"

#include <iostream>

int main() {
  try {
    winsock_session winsock_guard;  // ensures Winsock initialized on Windows
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
