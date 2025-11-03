#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace {
constexpr std::uint16_t kPort = 9000;
constexpr std::size_t kBufferSize = 1024;
}

int main() {
#ifdef _WIN32
    // TODO: Call WSAStartup(MAKEWORD(2, 2), &wsaData) and check the return value.
#endif

    // TODO: Create a TCP socket (AF_INET, SOCK_STREAM).
    // TODO: Set SO_REUSEADDR (and SO_EXCLUSIVEADDRUSE on Windows) where supported.
    // TODO: Bind to INADDR_ANY on kPort.
    // TODO: Listen with a backlog of at least 1.
    // TODO: Accept a single client connection.
    // TODO: Loop on recv/send to echo payloads until the client closes the socket.
    // TODO: Log timestamps for accepted connection and shutdown for post-analysis.

    // TODO: Close the accepted socket and the listening socket.

#ifdef _WIN32
    // TODO: Call WSACleanup() before exiting.
#endif
    return 0;
}
