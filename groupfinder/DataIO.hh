#ifndef DATAIO_HH
#define DATAIO_HH

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <hdf5/serial/H5Cpp.h>

namespace dataio {

using IDType = std::int64_t;

/**
 * @brief Structure to hold input observational data
 * Positions are assumed to be in spherical coords (radial distance, RA, Dec)
 */
struct ObsInputData {
    std::vector<double> masses;
    std::vector<std::array<double, 3>> positions; // 
    std::vector<IDType> ids;
    std::vector<double> velocities;
};

/**
 * @brief Structure to hold input simulation data
 * Positions are assumed to be in Cartesian coords (x, y, z)
 */
struct SimInputData {
    std::vector<double> masses;
    std::vector<std::array<double, 3>> positions;
    std::vector<IDType> ids;
    std::array<double, 3> ref_positions;
    std::array<double, 3> ref_velocities;
    std::vector<std::array<double, 3>> velocities;
};

/**
 * @brief Structure to hold concentration data for interpolation
 */
struct ConcentrationData {
    std::vector<double> halo_masses;
    std::vector<double> concentration;
};

/**
 * @brief Structure to hold group finder results
 */
struct GroupFinderResults {
    std::vector<std::vector<IDType>> group_member_ids;
    std::vector<IDType> central_ids;
    std::vector<double> halo_masses;
};

/**
 * @brief Structure to hold configuration parameters
 * These are just copied from the JSON config file but stored here
 * for redundancy.
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
    double box_size;
    double R_max;
    bool periodic;
    double B_scaling;
    double h;
};

/**
 * @brief Structure to hold statistics
 */

struct GroupFinderStatistics {
    std::int64_t total_groups; // total number of groups found
    std::int64_t isolated_count; // total number of isolated galaxies
    std::int64_t group_count; // total number of galaxies in groups (non-isolated)
};

/**
 * @brief HDF5 Data I/O Handler class
 */
class HDF5Handler {
public:
    /**
     * @brief Read observational input data from HDF5 file
     * @param filename Path to HDF5 file
     * @return ObsInputData structure
     */
    static ObsInputData readObsData(const std::string& filename);

    /**
     * @brief Read simulation input data from HDF5 file
     * @param filename Path to HDF5 file
     * @return SimInputData structure
     */
    static SimInputData readSimData(const std::string& filename);

    /**
     * @brief Read concentration data from HDF5 file
     * @param filename Path to HDF5 file
     * @return ConcentrationData structure
     */
    static ConcentrationData readConcentrationData(const std::string& filename);

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
        const std::string& timestamp = ""
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
     * @brief Read a 2D dataset from HDF5 file (for arrays)
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
     * @tparam T Data type
     * @param group HDF5 group
     * @param name Dataset name
     * @param data Data vector
     * @param predType HDF5 predefined data type
     */
    template<typename T>
    static void writeDataset1D(
        H5::Group& group,
        const std::string& name,
        const std::vector<T>& data,
        const H5::DataType& predType
    );

    /**
     * @brief Read a jagged 1D array (vector<vector<T>>) from HDF5
     * @tparam T Data type
     * @param dataset HDF5 dataset with variable-length datatype
     * @param predType HDF5 predefined data type
     * @return Vector<vector<T>>
     */
    template<typename T>
    static std::vector<std::vector<T>> readJagged1D(H5::DataSet& dataset, const H5::DataType& predType);

    /**
     * @brief Read a jagged 2D array (vector<vector<array<T,N>>>) from HDF5
     * @tparam T Data type
     * @tparam N Array size
     * @param dataset HDF5 dataset with variable-length datatype
     * @param predType HDF5 predefined data type
     * @return Vector<vector<array<T,N>>>
     */
    template<typename T, std::size_t N>
    static std::vector<std::vector<std::array<T, N>>> readJagged2D(H5::DataSet& dataset, const H5::DataType& predType);

    /**
     * @brief Write a jagged 1D array (vector<vector<T>>) to HDF5
     * @tparam T Data type
     * @param group HDF5 group
     * @param name Dataset name
     * @param data Data to write
     * @param predType HDF5 predefined data type
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