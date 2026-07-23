#ifndef DATAIO_HH
#define DATAIO_HH

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include "Types.hh"
#include <H5Cpp.h>

namespace dataio {

/**
 * @brief Input observational data
 * Positions are assumed to be in spherical coords (radial distance or z, RA, Dec)
 */
struct ObsInputData {
    std::vector<FloatType> masses;
    std::vector<std::array<FloatType, 3>> positions; // 
    std::vector<IDType> ids;
    std::vector<FloatType> velocities;
};

/**
 * @brief Input simulation data
 * Positions are assumed to be in Cartesian coords (x, y, z)
 */
struct SimInputData {
    std::vector<FloatType> masses;
    std::vector<std::array<FloatType, 3>> positions;
    std::vector<IDType> ids;
    std::array<FloatType, 3> ref_positions;
    std::array<FloatType, 3> ref_velocities;
    std::vector<std::array<FloatType, 3>> velocities;
};

/**
 * @brief Mass-concentration data for interpolation
 */
struct ConcentrationData {
    std::vector<double> halo_masses;
    std::vector<double> redshifts;
    std::vector<std::vector<double>> concentration;
};

struct SMHMInverseData {
    std::vector<double> log_mstar;
    std::vector<double> redshifts;
    std::vector<std::vector<double>> log_Mh;
};

struct RedshiftDistanceData {
    std::vector<double> redshifts;
    std::vector<double> distances;
};

struct GroupFinderResults {
    std::vector<IDType> central_ids;
    std::vector<FloatType> halo_masses;
    std::vector<IDType> group_member_ids;
    std::vector<IDType> group_member_offsets; // size n_groups + 1
};

/**
 * @brief Configuration parameters
 * These are just copied from the JSON config file but stored here for redundancy.
 */
struct GroupFinderConfig {
    int dim;
    bool obs;
    double R_h_group;
    double V_vir_group;
    double R_h_iso;
    double V_vir_iso;
    bool kdtree_search_used;
    bool satellite_reclassification_performed;
    bool isocentral_reclassification_performed;
    bool velocity_cut_imposed;
    bool contrast_val;
    bool use_comoving_distance;
    double box_size;
    double R_max;
    bool periodic;
    double B_scaling;
    double h;
    double omega_M;
};


struct GroupFinderStatistics {
    std::int64_t total_groups; // total number of groups found
    std::int64_t isolated_count; // total number of isolated galaxies
    std::int64_t group_count; // total number of galaxies in groups (non-isolated)
};

/**
 * @brief Data I/O class
 */
class HDF5Handler {
public:
    /**
     * @brief Read observational/simulation input data from HDF5 file
     * @param filename Path to HDF5 file
     * @return ObsInputData/SimInputData structure
     */
    static ObsInputData readObsData(const std::string& filename);
    static SimInputData readSimData(const std::string& filename);

    /**
     * @brief Read input interpolation data from their HDF5 files
     */
    static ConcentrationData readConcentrationData(const std::string& filename);
    static SMHMInverseData readSMHMInverseData(const std::string& filename);
    static RedshiftDistanceData readRedshiftDistanceData(const std::string& filename);

    /**
     * @brief Write group finder results to HDF5 file
     * @param filename Output HDF5 file path
     * @param results Observational/simulation group finder results
     * @param config Configuration parameters
     * @param stats Statistics
     * @param timestamp Optional timestamp string
     */
    static void writeResults(
        const std::string& filename,
        const GroupFinderResults& results,
        const GroupFinderConfig& config,
        const GroupFinderStatistics& stats,
        const std::string& timestamp = "",
        bool chunk = false, hsize_t chunk_size = 1000000
    );

private:
    /**
     * @brief Read a 1D dataset from HDF5 file
     * @tparam T Data type
     * @param dataset HDF5 dataset
     * @param predType HDF5 predefined data type
     * @return Vector of data
     */
    template<typename T>
    static std::vector<T> readDataset1D(H5::DataSet& dataset, const H5::DataType& predType);

    /**
     * @brief Read a 2D dataset from HDF5 file (for vecs of arrays)
     * @tparam T Data type
     * @tparam N Array size
     * @param dataset HDF5 dataset
     * @param predType HDF5 predefined data type
     * @return Vector of arrays
    */
    template<typename T, std::size_t N>
    static std::vector<std::array<T, N>> readDataset2D(H5::DataSet& dataset, const H5::DataType& predType);

    /**
     * @brief Write a 1D dataset to HDF5 file
     */
    template<typename T>
    static void writeDataset1D(
        H5::Group& group,
        const std::string& name,
        const std::vector<T>& data,
        const H5::DataType& predType,
        bool chunk = false, 
        hsize_t chunk_size = 1000000
    );

    /**
     * @brief Read a jagged 1D array (vector<vector<T>>) from HDF5
    */
    template<typename T>
    static std::vector<std::vector<T>> readJagged1D(H5::DataSet& dataset, const H5::DataType& predType);

    /**
     * @brief Read a jagged 2D array (vector<vector<array<T,N>>>) from HDF5
    */
    template<typename T, std::size_t N>
    static std::vector<std::vector<std::array<T, N>>> readJagged2D(H5::DataSet& dataset, const H5::DataType& predType);

    /**
     * @brief Write a jagged 1D array (vector<vector<T>>) to HDF5
    */
    template<typename T>
    static void writeJagged1D(
        H5::Group& group,
        const std::string& name,
        const std::vector<std::vector<T>>& data,
        const H5::DataType& predType
    );
};

} // namespace dataio

#endif // DATAIO_HH
