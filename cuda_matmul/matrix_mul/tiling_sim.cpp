#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

struct TileResult {
    std::size_t tiles_m;
    std::size_t tiles_n;
    std::size_t total_tiles;
    double useful_fraction;
};

TileResult analyze_tiles(double M, double N, double Mtile, double Ntile) {
    const std::size_t tiles_m = static_cast<std::size_t>(std::ceil(M / Mtile));
    const std::size_t tiles_n = static_cast<std::size_t>(std::ceil(N / Ntile));
    const std::size_t total_tiles = tiles_m * tiles_n;

    const double useful = M * N;
    const double tile_work = static_cast<double>(total_tiles) * Mtile * Ntile;
    const double useful_fraction = useful / tile_work;

    return {tiles_m, tiles_n, total_tiles, useful_fraction};
}

void sweep_over_N(double M, double N_start, double N_end, double step, double Mtile, double Ntile) {
    std::cout << "\nSweep over N to observe tile quantization:" << '\n';
    std::cout << "N\tTiles\tUseful%" << '\n';
    for (double N = N_start; N <= N_end + 1e-9; N += step) {
        const auto res = analyze_tiles(M, N, Mtile, Ntile);
        std::cout << std::setw(5) << static_cast<int>(N) << '\t'
                  << std::setw(5) << res.total_tiles << '\t'
                  << std::fixed << std::setprecision(2) << res.useful_fraction * 100.0 << '\n';
    }
}

int main() {
    double M = 0.0, N = 0.0;
    double Mtile = 128.0, Ntile = 128.0;

    std::cout << "Tile Quantization Simulator\n";
    std::cout << "----------------------------\n";

    std::cout << "Enter M (rows): ";
    std::cin >> M;
    std::cout << "Enter N (columns): ";
    std::cin >> N;
    std::cout << "Enter tile height Mtile (default 128): ";
    std::cin >> Mtile;
    std::cout << "Enter tile width Ntile (default 128): ";
    std::cin >> Ntile;

    const auto res = analyze_tiles(M, N, Mtile, Ntile);
    std::cout << '\n';
    std::cout << "Tiles along M: " << res.tiles_m << '\n';
    std::cout << "Tiles along N: " << res.tiles_n << '\n';
    std::cout << "Total tiles : " << res.total_tiles << '\n';
    std::cout << std::fixed << std::setprecision(2)
              << "Useful work : " << res.useful_fraction * 100.0 << "%\n";
    std::cout << "Wasted work : " << (1.0 - res.useful_fraction) * 100.0 << "%\n";

    char sweep_choice = 'n';
    std::cout << "\nSweep N to see quantization steps? (y/n): ";
    std::cin >> sweep_choice;

    if (sweep_choice == 'y' || sweep_choice == 'Y') {
        double N_start = std::max(1.0, N - Ntile);
        double N_end = N + Ntile;
        double step = Ntile / 8.0;
        sweep_over_N(M, N_start, N_end, step, Mtile, Ntile);
    }

    return 0;
}