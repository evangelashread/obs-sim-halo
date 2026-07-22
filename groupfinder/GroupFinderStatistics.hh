#include "Types.hh"
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <cassert>
#include <cstdint>

double Median(std::vector<double> data);

double MAD(const std::vector<double>& data);

std::tuple<std::vector<double>, std::vector<double>> number_counts(
    const std::vector<double>& arr, const std::vector<double>& massbins,
    const double& vol, bool cumulative, bool mf, bool no_edges = false
);

std::tuple<std::vector<IDType>, std::vector<IDType>> divide_groups(
    const std::vector<std::vector<IDType>>& group_indices, 
    const std::vector<IDType>& central_ids);

std::vector<IDType> intersection(std::vector<IDType>& v1, std::vector<IDType>& v2);