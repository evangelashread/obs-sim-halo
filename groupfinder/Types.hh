#pragma once
#include <cstdint>
#include <array>
#include <H5Cpp.h>

// Single switch controlling the storage precision used for all per galaxy point data,
// both float AND int (IDs, positions, velocities, masses, halo properties, etc.) 
// across both the group finder core and the HDF5 I/O layer.

// Only ever need to change these two lines
using FloatType = float;
using IDType = std::int64_t;
using Vec3 = std::array<FloatType,3>;

template<typename T> struct HDF5FloatType;
template<> struct HDF5FloatType<float>  { static const H5::PredType& value() { return H5::PredType::NATIVE_FLOAT; } };
template<> struct HDF5FloatType<double> { static const H5::PredType& value() { return H5::PredType::NATIVE_DOUBLE; } };

template<typename T> struct HDF5IntType;
template<> struct HDF5IntType<int> { static const H5::PredType& value() { return H5::PredType::NATIVE_INT; } };
template<> struct HDF5IntType<std::int64_t> { static const H5::PredType& value() { return H5::PredType::NATIVE_INT64; } };