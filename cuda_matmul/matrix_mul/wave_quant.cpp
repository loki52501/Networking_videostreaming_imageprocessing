#include <cmath>
#include <iomanip>
#include <iostream>

struct WaveStats {
    std::size_t tiles_total;
    std::size_t tiles_per_wave;
    std::size_t full_waves;
    std::size_t tail_tiles;
    double tail_occupancy;
};

WaveStats analyze_wave(double M, double N, double Mtile, double Ntile, std::size_t sms, std::size_t tiles_per_sm) {
    const std::size_t tiles_m = static_cast<std::size_t>(std::ceil(M / Mtile));
    const std::size_t tiles_n = static_cast<std::size_t>(std::ceil(N / Ntile));
    const std::size_t tiles_total = tiles_m * tiles_n;

    const std::size_t tiles_per_wave = sms * tiles_per_sm;
    const std::size_t full_waves = tiles_total / tiles_per_wave;
    const std::size_t tail_tiles = tiles_total % tiles_per_wave;
    const double tail_occupancy = tiles_per_wave == 0 ? 0.0 : static_cast<double>(tail_tiles) / static_cast<double>(tiles_per_wave);

    return {tiles_total, tiles_per_wave, full_waves, tail_tiles, tail_occupancy};
}

int main() {
    double M = 0.0, N = 0.0;
    double Mtile = 128.0, Ntile = 128.0;
    std::size_t sms = 48;         // RTX 2050 has 16 SM? adjust via input
    std::size_t tiles_per_sm = 1; // default assumption

    std::cout << "Wave Quantization Playground\n";
    std::cout << "----------------------------\n";

    std::cout << "Enter M (rows): ";
    std::cin >> M;
    std::cout << "Enter N (columns): ";
    std::cin >> N;
    std::cout << "Enter tile height Mtile: ";
    std::cin >> Mtile;
    std::cout << "Enter tile width Ntile: ";
    std::cin >> Ntile;
    std::cout << "Enter SM count (e.g., 16 for RTX 2050): ";
    std::cin >> sms;
    std::cout << "Tiles per SM (concurrent thread blocks per SM): ";
    std::cin >> tiles_per_sm;

    const auto stats = analyze_wave(M, N, Mtile, Ntile, sms, tiles_per_sm);

    std::cout << '\n'
              << "Total tiles            : " << stats.tiles_total << '\n'
              << "Tiles per wave          : " << stats.tiles_per_wave << '\n'
              << "Full waves              : " << stats.full_waves << '\n'
              << "Tail tiles              : " << stats.tail_tiles << '\n'
              << std::fixed << std::setprecision(2)
              << "Tail occupancy (SM %)   : " << stats.tail_occupancy * 100.0 << "\n"
              << "Predicted throughput%   : " << (stats.full_waves + stats.tail_occupancy) / (stats.full_waves + (stats.tail_tiles > 0 ? 1.0 : 0.0)) * 100.0 << '\n';

    std::cout << "\nTip: try N values near multiples of tile*SM to see occupancy drops." << '\n';
    return 0;
}