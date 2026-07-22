#include "GroupFinderStatistics.hh"
#include "Types.hh"
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <cassert>
#include <cstdint>
#include <iterator>

/* 
Module for statistical calculations in C++.

Intended to be used for calculating the mass function by group/field classification.
Can be used for calculating the median or median absolute deviation of group and field counts
per mass bin across multiple Monte Carlo realizations of the group finder.
*/

double Median(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    size_t size = data.size();
    if (size % 2 == 0) { // Handles case of even number of points
        return (data[size / 2 - 1] + data[size / 2]) / 2.0;
    } else if (size == 1) {
        return data[0]; // If only one element, return it
    } else {
        return data[size / 2];
    }
}

// Function to calculate the MAD
double MAD(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0; // Return 0 for empty data
    }
    double median = Median(data);
    std::vector<double> deviations;
    for (double x : data) {
        deviations.push_back(std::abs(x - median));
    }
    return Median(deviations);
}

// Find the number of galaxies in each mass bin
std::tuple<std::vector<double>, std::vector<double>> number_counts(
    const std::vector<double>& arr, const std::vector<double>& massbins,
    const double& vol, bool cumulative, bool mf, bool no_edges
) {
    std::vector<double> number_arr(massbins.size()-1);
    std::vector<double> binvals(massbins.size()-1);
    for (size_t i = 0; i < massbins.size()-1; ++i) {
        double binned_vals = 0.;
        double bin_width = 0.;
        if (no_edges) {
            if (i == 0) {
                if (cumulative) {
                    for (size_t j = 0; j < arr.size(); ++j) {
                        binned_vals += 1.;
                    }
                } else {
                    if (mf) bin_width = massbins[i+1] - *std::min_element(arr.begin(), arr.end());
                    for (size_t j = 0; j < arr.size(); ++j) {
                        if (arr[j] < massbins[i+1]) {
                            binned_vals += 1.;
                        }
                    }
                }
            } else if (i == massbins.size()-2) {
                if (mf && !cumulative) bin_width = *std::max_element(arr.begin(), arr.end()) - massbins[i];
                for (size_t j = 0; j < arr.size(); ++j) {
                    if (arr[j] >= massbins[i]) {
                        binned_vals += 1.;
                    }
                }
            } else {
                if (cumulative) {
                    for (size_t j = 0; j < arr.size(); ++j) {
                        if (arr[j] >= massbins[i]) {
                            binned_vals += 1.;
                        }
                    }
                } else {
                    if (mf) bin_width = massbins[i+1] - massbins[i];
                    for (size_t j = 0; j < arr.size(); ++j) {
                        if (arr[j] >= massbins[i] && arr[j] < massbins[i+1]) {
                            binned_vals += 1.;
                        }
                    }
                }
            }
        } else {
            if (cumulative) {
                for (size_t j = 0; j < arr.size(); ++j) {
                    if (arr[j] >= massbins[i] && arr[j] <= massbins[massbins.size()-1]) {
                        binned_vals += 1.;
                    }
                }
            } else {
                if (mf) bin_width = massbins[i+1] - massbins[i];
                for (size_t j = 0; j < arr.size(); ++j) {
                    if (arr[j] >= massbins[i] && arr[j] < massbins[i+1]) {
                        binned_vals += 1.;
                    }
                }
            }
        }
        if (mf) {
            if (cumulative) {
                number_arr[i] = binned_vals / vol;
            } else {
                number_arr[i] = binned_vals / (vol * bin_width);
            }
        } else number_arr[i] = binned_vals;
        double bincenter = 0.;
        if (cumulative) {
            bincenter = massbins[i];
        } else {
            bincenter = 0.5*(massbins[i] + massbins[i+1]);
        }
        binvals[i] = bincenter;
    }
    return std::make_tuple(number_arr, binvals);
}

// Function to divide into group and field classification based on indices
std::tuple<std::vector<IDType>, std::vector<IDType>> divide_groups(
    const std::vector<std::vector<IDType>>& group_indices, 
    const std::vector<IDType>& central_ids) {

    std::vector<IDType> field_galaxies;
    std::vector<IDType> group_galaxies;

    field_galaxies.reserve(central_ids.size());
    group_galaxies.reserve(5*central_ids.size()); // Rough estimate

    for (size_t i = 0; i < group_indices.size(); ++i) {
        assert(group_indices[i][0] == central_ids[i]); // Ensure central is first
        if (group_indices[i].size() == 1) {
            field_galaxies.push_back(group_indices[i][0]);
        } else {
            group_galaxies.insert(group_galaxies.end(), group_indices[i].begin(), group_indices[i].end());
        }
    }
    return std::make_tuple(field_galaxies, group_galaxies);
}

// Simple function to find the intersection of two vectors
std::vector<IDType> intersection(std::vector<IDType>& v1, std::vector<IDType>& v2) {
    std::vector<IDType> v3;
    std::stable_sort(v1.begin(), v1.end());
    std::stable_sort(v2.begin(), v2.end());
    std::set_intersection(v1.begin(),v1.end(),
                        v2.begin(),v2.end(),
                        std::back_inserter(v3));
    return v3;
}

// TODO: Add more statistical functions as needed