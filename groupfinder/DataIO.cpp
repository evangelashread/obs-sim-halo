#include "DataIO.hh"
#include <cstring>
#include <hdf5/serial/H5Cpp.h>

namespace dataio {

// Template implementations for reading 1D datasets
// e.g. masses in run_groupfinder_obs.cpp
template<typename T>
std::vector<T> HDF5Handler::readDataset1D(H5::DataSet& dataset, const H5::DataType& predType) {
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, nullptr);
    
    std::vector<T> data(dims[0]);
    dataset.read(data.data(), predType);
    return data;
}

// Template implementations for reading 2D datasets (for arrays)
// e.g. MW positions in run_groupfinder_sim.cpp, or RA/Dec positions in run_groupfinder_obs.cpp
template<typename T, std::size_t N>
std::vector<std::array<T, N>> HDF5Handler::readDataset2D(H5::DataSet& dataset, const H5::DataType& predType) {
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[2];
    int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    
    if (ndims != 2 || dims[1] != N) {
        throw std::runtime_error("Dataset dimensions do not match expected array size");
    }
    
    std::vector<std::array<T, N>> data(dims[0]);
    dataset.read(data.data(), predType);
    return data;
}

// Template implementations for writing 1D datasets
// e.g. central ids in run_groupfinder_obs.cpp
template<typename T>
void HDF5Handler::writeDataset1D(
    H5::Group& group,
    const std::string& name,
    const std::vector<T>& data,
    const H5::DataType& predType
) {
    hsize_t dims[1] = {data.size()};
    H5::DataSpace dataspace(1, dims);
    H5::DataSet dataset = group.createDataSet(name, predType, dataspace);
    dataset.write(data.data(), predType);
}

// Read jagged 1D array (vector<vector<T>>)
// e.g. masses in run_completeness_sim.cpp
template<typename T>
std::vector<std::vector<T>> HDF5Handler::readJagged1D(
    H5::DataSet& dataset, const H5::DataType& predType) {
    
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, nullptr);
    
    // Create variable-length datatype
    H5::VarLenType vl_type(&predType);
    
    // Read variable-length data
    std::vector<hvl_t> vl_data(dims[0]);
    dataset.read(vl_data.data(), vl_type);
    
    // Convert to vector<vector<T>>
    std::vector<std::vector<T>> result(dims[0]);
    for (size_t i = 0; i < dims[0]; ++i) {
        result[i].resize(vl_data[i].len);
        std::memcpy(result[i].data(), vl_data[i].p, vl_data[i].len * sizeof(T));
    }
    
    // Free variable-length memory
    dataset.vlenReclaim(vl_data.data(), vl_type, dataspace);
    
    return result;
}

// Read jagged 2D array (vector<vector<array<T,N>>>)
// e.g. the positions and velocities in run_groupfinder_sim.cpp
template<typename T, std::size_t N>
std::vector<std::vector<std::array<T, N>>> HDF5Handler::readJagged2D(
    H5::DataSet& dataset, const H5::DataType& predType) {
    
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, nullptr);
    
    // Create array type and variable-length wrapper
    hsize_t array_dims[1] = {N};
    H5::ArrayType array_type(predType, 1, array_dims);
    H5::VarLenType vl_type(&array_type);
    
    // Read variable-length data
    std::vector<hvl_t> vl_data(dims[0]);
    dataset.read(vl_data.data(), vl_type);
    
    // Convert to vector<vector<array<T,N>>>
    std::vector<std::vector<std::array<T, N>>> result(dims[0]);
    for (size_t i = 0; i < dims[0]; ++i) {
        size_t n_elements = vl_data[i].len;
        result[i].resize(n_elements);
        std::memcpy(result[i].data(), vl_data[i].p, n_elements * sizeof(std::array<T, N>));
    }
    
    // Free variable-length memory
    dataset.vlenReclaim(vl_data.data(), vl_type, dataspace);
    
    return result;
}

// Write jagged 1D array (vector<vector<T>>)
// e.g. vector of central IDs for some number of MWs
template<typename T>
void HDF5Handler::writeJagged1D(
    H5::Group& group,
    const std::string& name,
    const std::vector<std::vector<T>>& data,
    const H5::DataType& predType) {
    
    // Create variable-length datatype
    H5::VarLenType vl_type(&predType);
    
    // Create dataspace for outer vector
    hsize_t dims[1] = {data.size()};
    H5::DataSpace dataspace(1, dims);
    
    // Create dataset
    H5::DataSet dataset = group.createDataSet(name, vl_type, dataspace);
    
    // Prepare variable-length data
    std::vector<hvl_t> vl_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        vl_data[i].len = data[i].size();
        vl_data[i].p = const_cast<T*>(data[i].data());
    }
    
    // Write data
    dataset.write(vl_data.data(), vl_type);
    
    // Add metadata attribute
    H5::Attribute attr = dataset.createAttribute("outer_size", H5::PredType::NATIVE_UINT64, 
                                                  H5::DataSpace(H5S_SCALAR));
    uint64_t outer_size = data.size();
    attr.write(H5::PredType::NATIVE_UINT64, &outer_size);
}

ObsInputData HDF5Handler::readObsData(const std::string& filename) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        ObsInputData data;
        
        // Read masses
        {
            H5::DataSet dataset = file.openDataSet("masses");
            data.masses = readDataset1D<double>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        // Read positions (as 2D array)
        {
            H5::DataSet dataset = file.openDataSet("positions");
            data.positions = readDataset2D<double, 3>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        // Read ids
        {
            H5::DataSet dataset = file.openDataSet("ids");
            data.ids = readDataset1D<std::int64_t>(dataset, H5::PredType::NATIVE_INT64);
        }
        
        // Read velocities
        {
            H5::DataSet dataset = file.openDataSet("velocities");
            data.velocities = readDataset1D<double>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        file.close();
        std::cout << "Data loaded successfully from HDF5 file: " << filename << std::endl;
        
        return data;
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("HDF5 error reading galaxy data: " + std::string(e.getCDetailMsg()));
    }
}

SimInputData HDF5Handler::readSimData(const std::string& filename) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        SimInputData data;
        
        // Read masses (as jagged 1D array)
        {
            H5::DataSet dataset = file.openDataSet("masses");
            data.masses = readDataset1D<double>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        // Read positions (as jagged 2D array)
        {
            H5::DataSet dataset = file.openDataSet("positions");
            data.positions = readDataset2D<double, 3>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        // Read MW positions (single reference point from 2D array)
        {
            H5::DataSet dataset = file.openDataSet("ref_positions");
            auto ref_pos_vec = readDataset2D<double, 3>(dataset, H5::PredType::NATIVE_DOUBLE);
            if (ref_pos_vec.empty()) {
                throw std::runtime_error("ref_positions dataset is empty");
            }
            data.ref_positions = ref_pos_vec[0];
        }

        // Read MW velocities (single reference point from 2D array)
        {
            H5::DataSet dataset = file.openDataSet("ref_velocities");
            auto ref_vel_vec = readDataset2D<double, 3>(dataset, H5::PredType::NATIVE_DOUBLE);
            if (ref_vel_vec.empty()) {
                throw std::runtime_error("ref_velocities dataset is empty");
            }
            data.ref_velocities = ref_vel_vec[0];
        }
        
        // Read ids (as 1D array)
        {
            H5::DataSet dataset = file.openDataSet("ids");
            data.ids = readDataset1D<std::int64_t>(dataset, H5::PredType::NATIVE_INT64);
        }
        
        // Read velocities (as 2D array)
        {
            H5::DataSet dataset = file.openDataSet("velocities");
            data.velocities = readDataset2D<double, 3>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        file.close();
        std::cout << "Data loaded successfully from HDF5 file: " << filename << std::endl;
        
        return data;
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("HDF5 error reading galaxy data: " + std::string(e.getCDetailMsg()));
    }
}

ConcentrationData HDF5Handler::readConcentrationData(const std::string& filename) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        ConcentrationData data;
        
        // Read halo_masses
        {
            H5::DataSet dataset = file.openDataSet("halo_masses");
            data.halo_masses = readDataset1D<double>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        // Read concentration
        {
            H5::DataSet dataset = file.openDataSet("concentration");
            data.concentration = readDataset1D<double>(dataset, H5::PredType::NATIVE_DOUBLE);
        }
        
        file.close();
        std::cout << "Concentration data loaded successfully from: " << filename << std::endl;
        
        return data;
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("HDF5 error reading concentration data: " + std::string(e.getCDetailMsg()));
    }
}

// Function to write results
void HDF5Handler::writeResults(
    const std::string& filename,
    const GroupFinderResults& results,
    const GroupFinderConfig& config,
    const GroupFinderStatistics& stats,
    const std::string& timestamp
) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        
        // Create root group for results
        H5::Group results_group(file.createGroup("/results"));
        
        // Write central_ids
        writeDataset1D(results_group, "central_ids", results.central_ids, H5::PredType::NATIVE_INT64);
            
        // Write halo_masses
        writeDataset1D(results_group, "halo_masses", results.halo_masses, H5::PredType::NATIVE_DOUBLE);
            
        // Write flattened group member IDs
        writeJagged1D(results_group, "group_member_ids", results.group_member_ids, H5::PredType::NATIVE_INT64);
        
        // Create configuration group
        H5::Group config_group(file.createGroup("/configuration"));

        // Create selection criteria dataset
        {
            double sel_data[] = {config.R_h_group, config.V_vir_group, config.R_h_iso, config.V_vir_iso};
            hsize_t dims[1] = {4};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = config_group.createDataSet("selection_criteria", 
                                                              H5::PredType::NATIVE_DOUBLE, 
                                                              dataspace);
            dataset.write(sel_data, H5::PredType::NATIVE_DOUBLE);
            
            // Add named attributes to explain the data
            H5::StrType str_type(H5::PredType::C_S1, 256);
            H5::Attribute attr_names = dataset.createAttribute("names", str_type, 
                                                                H5::DataSpace(H5S_SCALAR));
            attr_names.write(str_type, std::string("R_h_group, V_vir_group, R_h_iso, V_vir_iso"));
        }
        
        // Write boolean settings as attributes
        config_group.createAttribute("kdtree_search_used", 
                                    H5::PredType::NATIVE_HBOOL, 
                                    H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HBOOL, 
                                                                      &config.kdtree_search_used);
        config_group.createAttribute("satellite_reclassification_performed", 
                                    H5::PredType::NATIVE_HBOOL, 
                                    H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HBOOL, 
                                                                      &config.satellite_reclassification_performed);
        config_group.createAttribute("isocentral_reclassification_performed", 
                                    H5::PredType::NATIVE_HBOOL, 
                                    H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HBOOL, 
                                                                      &config.isocentral_reclassification_performed);
        config_group.createAttribute("velocity_cut_imposed", 
                                    H5::PredType::NATIVE_HBOOL, 
                                    H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HBOOL, 
                                                                      &config.velocity_cut_imposed);
        
        // Write individual selection criteria values as attributes
        config_group.createAttribute("R_h_group", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.R_h_group);
        config_group.createAttribute("V_vir_group", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.V_vir_group);
        config_group.createAttribute("R_h_iso", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.R_h_iso);
        config_group.createAttribute("V_vir_iso", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.V_vir_iso);

        config_group.createAttribute("density_contrast", H5::PredType::NATIVE_HBOOL, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HBOOL, 
                                                                       &config.contrast_val);
        config_group.createAttribute("box_size", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.box_size);
        config_group.createAttribute("periodic", H5::PredType::NATIVE_HBOOL, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HBOOL, 
                                                                       &config.periodic);
        config_group.createAttribute("B_scaling", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.B_scaling);
        config_group.createAttribute("h", H5::PredType::NATIVE_DOUBLE, 
                                     H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_DOUBLE, 
                                                                       &config.h);
        
        // Create statistics group
        H5::Group stats_group(file.createGroup("/statistics"));

        std::int64_t stats_data[] = {stats.total_groups, stats.isolated_count, stats.group_count};
        hsize_t dims[1] = {3};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = stats_group.createDataSet("group_stats", 
                                                            H5::PredType::NATIVE_INT64, 
                                                            dataspace);
        dataset.write(stats_data, H5::PredType::NATIVE_INT64);
            
        H5::StrType str_type(H5::PredType::C_S1, 256);
        H5::Attribute attr_names = dataset.createAttribute("names", str_type, 
                                                                H5::DataSpace(H5S_SCALAR));
        attr_names.write(str_type, std::string("total_groups, isolated_count, group_count"));
        
        // Write timestamp as attribute on root if provided
        if (!timestamp.empty()) {
            H5::StrType str_type(H5::PredType::C_S1, 256);
            file.createAttribute("timestamp", str_type, H5::DataSpace(H5S_SCALAR)).write(str_type, timestamp);
        }
        
        file.close();
        
        std::cout << "Wrote HDF5 output: " << filename << "\n";
        std::cout << "File structure:\n";
        std::cout << "  /results/\n";
        std::cout << "    - central_ids\n";
        std::cout << "    - halo_masses\n";
        std::cout << "    - group_member_ids\n";
        std::cout << "  /configuration/\n";
        std::cout << "    - selection_criteria\n";
        std::cout << "    - (attributes: kdtree_search_used, satellite_reclassification_performed, ...)\n";
        std::cout << "  /statistics/\n";
        std::cout << "    - group_stats\n";
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("HDF5 error writing results: " + std::string(e.getCDetailMsg()));
    }
}

} // namespace dataio