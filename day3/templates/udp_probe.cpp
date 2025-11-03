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
#include <iostream>
#include <string>
#include <thread>

namespace {
constexpr std::uint16_t kPort = 9100;
constexpr const char* kPeerAddress = "127.0.0.1";
constexpr std::size_t kDatagramCount = 200;
constexpr const char* kLogPath = "../logs/udp_jitter.txt";
}

struct PacketRecord {
    std::uint32_t sequence_id{};
    std::chrono::microseconds send_ts{};
    std::chrono::microseconds recv_ts{};
};

int main() {
#ifdef _WIN32
    // TODO: Initialise Winsock.
#endif

    // TODO: Create a UDP socket (AF_INET, SOCK_DGRAM).
    // TODO (optional): Bind to a local port if you need to receive on the same process.
    // TODO: Build payloads that include sequence IDs and timestamps.
    // TODO: Send kDatagramCount packets spaced by a configurable interval.
    // TODO: Receive echoes (or run a companion listener) to detect loss/reordering.
    // TODO: Persist sequence, send_ts, recv_ts, and gap analysis to kLogPath.

    // TODO: Output summary stats (drops, jitter estimates) to stdout.

#ifdef _WIN32
    // TODO: Clean up socket and WSACleanup().
#else
    // TODO: Close the socket.
#endif
    return 0;
}
