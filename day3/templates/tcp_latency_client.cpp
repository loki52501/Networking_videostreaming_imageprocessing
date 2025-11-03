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

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr std::uint16_t kPort = 9000;
constexpr const char* kServerAddress = "127.0.0.1";
constexpr std::size_t kIterations = 50;
constexpr const char* kLogPath = "../logs/rtt_baseline.txt";
}

int main() {
#ifdef _WIN32
    // TODO: Invoke WSAStartup(MAKEWORD(2, 2), &wsaData) and ensure it succeeds.
#endif

    // TODO: Create a TCP socket (AF_INET, SOCK_STREAM).
    // TODO: Populate sockaddr_in with kServerAddress and kPort (inet_pton or InetPton).
    // TODO: Connect to the server and handle connection errors.

    // TODO: Open std::ofstream log_file(kLogPath, std::ios::out | std::ios::trunc).
    // TODO: For each iteration:
    //   1. Record start time (std::chrono::high_resolution_clock).
    //   2. Send a payload (e.g., "ping" plus sequence id).
    //   3. Receive the echo.
    //   4. Record end time and compute RTT in milliseconds.
    //   5. Write iteration number and RTT to log_file.
    //   6. Sleep for ~100ms between probes (std::this_thread::sleep_for).

    // TODO: Flush and close the log file.
    // TODO: Close the socket handle.

#ifdef _WIN32
    // TODO: Call WSACleanup() here as well.
#endif
    return 0;
}
