#ifndef TIMING
#define TIMING
#include <chrono>
#include<vector>

namespace timing {
void randomvalue(std::vector<std::vector<int>>& A,
              int row, int column);
std::vector<std::vector<int>> matrixadd(std::vector<std::vector<int>>& A,
              std::vector<std::vector<int>>& B);

std::vector<std::vector<int>>  matrixsub(std::vector<std::vector<int>>& A,
              std::vector<std::vector<int>>& B);

std::vector<std::vector<int>>  matrixmul(std::vector<std::vector<int>>& A,
              std::vector<std::vector<int>>& B);

void printes(std::vector<std::vector<int>>& A);
}  // namespace timing

#endif  // TIMING_HPP