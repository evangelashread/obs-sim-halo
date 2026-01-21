
#include "GroupFinderCore.hh"
#include "DataIO.hh"
#include <iostream>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iomanip>
#include <iterator>
#include <cstdint>
#include <stdexcept>
#include <sstream>

bool stobool(const std::string& s) {
    std::istringstream iss(s);
    bool b;
    if (!(iss >> std::boolalpha >> b))
        throw std::invalid_argument("stobool: invalid input");
    return b;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_files> <output_files> <config_file>\n";
        return 1;
    }
    
    // Parse comma-separated input and output filenames
    std::vector<std::string> infilenames;
    std::vector<std::string> outfilenames;
    
    std::stringstream instream(argv[1]);
    std::string infile;
    while (std::getline(instream, infile, ',')) {
        infilenames.push_back(infile);
        std::cout << "Read input file: " << infile << std::endl;
    }
    
    std::stringstream outstream(argv[2]);
    std::string outfile;
    while (std::getline(outstream, outfile, ',')) {
        outfilenames.push_back(outfile);
        std::cout << "Read output file: " << outfile << std::endl;
    }
    
    const std::string config_file = argv[3];

    using namespace gf;
    using json = nlohmann::json;
    using IDType = std::int64_t;

    // Get time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    std::stringstream filetime;
    filetime << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S");

    // Load concentration data using HDF5 handler
    dataio::ConcentrationData conc_data;
    try {
        conc_data = dataio::HDF5Handler::readConcentrationData("input/concentration.h5");
    } catch (const std::exception& e) {
        std::cerr << "Error loading concentration data: " << e.what() << "\n";
        return 1;
    }

    // Configure run parameters: read in from config JSON file
    std::ifstream config_stream(config_file);
    if (!config_stream) {
        std::cerr << "Cannot open config file: " << config_file << "\n";
        return 1;
    }
    json config_json;
    config_stream >> config_json;

    bool obs_val = config_json["obs"].get<bool>(); // false for simulation run
    double R_h_group_val = config_json["R_h_group"].get<double>(); 
    double V_vir_group_val = config_json["V_vir_group"].get<double>();
    double R_h_iso_val = config_json["R_h_iso"].get<double>(); 
    double V_vir_iso_val = config_json["V_vir_iso"].get<double>();
    bool vel_cut_val = config_json["vel_cut"].get<bool>();
    bool tree_search_val = config_json["tree_search"].get<bool>();
    bool sat_reclass_val = config_json["sat_reclass"].get<bool>();
    bool iso_reclass_val = config_json["iso_reclass"].get<bool>();
    bool contrast_val = config_json["contrast"].get<bool>();
    double R_max = config_json["R_max"].get<double>();
    double box_size = config_json["box_size"].get<double>();
    bool periodic = config_json["periodic"].get<bool>();
    double B_scaling = config_json["B_scaling"].get<double>();
    double h_val = config_json["h"].get<double>();
    int dim = config_json["dim"].get<int>(); // 3 or 6 for simulation run
    
    SelectionCriteria sel{
        R_h_group_val,
        V_vir_group_val,
        R_h_iso_val,
        V_vir_iso_val
    };
    
    GroupFinderSettings config{
        obs_val,
        vel_cut_val,
        tree_search_val,
        sat_reclass_val,
        iso_reclass_val,
        contrast_val
    };

    // Run for each input file
    for (size_t ref = 0; ref < infilenames.size(); ++ref) {
        dataio::SimInputData galaxy_data;
        try {
                galaxy_data = dataio::HDF5Handler::readSimData(infilenames[ref]);
            } catch (const std::exception& e) {
                std::cerr << "Error loading galaxy data: " << e.what() << "\n";
                return 1;
            }

        std::ofstream outfile(outfilenames[ref]);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            return 1;
        }

        std::vector<std::vector<IDType>> group_indices;
        std::vector<IDType> central_ids;
        std::vector<double> halo_masses;
        
        if (dim == 6) {
            GroupFinder<Dist3D, VelPeculiar3D> finder(sel, config, box_size, h_val*100., periodic);

            finder.set_conc_table(conc_data.halo_masses, conc_data.concentration);

            auto result = finder.run_once(galaxy_data.masses, galaxy_data.ids, galaxy_data.positions, 
                    galaxy_data.velocities, galaxy_data.ref_positions, galaxy_data.ref_velocities, 
                    R_max, B_scaling, periodic);

            group_indices = std::get<0>(result);
            central_ids = std::get<1>(result);
            halo_masses = std::get<2>(result);

        } else if (dim == 3) {
            GroupFinder<DistProjectedRCTan, VelTotal> finder(sel, config, box_size, h_val*100., periodic);

            finder.set_conc_table(conc_data.halo_masses, conc_data.concentration);

            auto result = finder.run_once(galaxy_data.masses, galaxy_data.ids, galaxy_data.positions, 
                    galaxy_data.velocities, galaxy_data.ref_positions, galaxy_data.ref_velocities, 
                    R_max, B_scaling, periodic);

            group_indices = std::get<0>(result);
            central_ids = std::get<1>(result);
            halo_masses = std::get<2>(result);
        } else {
            std::cerr << "Invalid dimension specified in config file. Use 3 or 6." << std::endl;
            return 1;
        }

        // Add group statistics
        std::int64_t total_groups = 0;
        std::int64_t isolated_count = 0;
        std::int64_t group_count = 0;

        for (const auto& group : group_indices) {
            total_groups++;
            if (group.size() == 1) {
                isolated_count++;
            } else {
                group_count += group.size();
            }
        }
        std::cout << "MW " << (ref+1) << ": Total groups = " << total_groups
                    << ", Isolated centrals = " << isolated_count
                    << ", Group members = " << group_count << std::endl;

        // Prepare results structure
        dataio::GroupFinderResults results;
        results.group_member_ids = group_indices;
        results.central_ids = central_ids;
        results.halo_masses = halo_masses;
        
        // Prepare configuration structure
        dataio::GroupFinderConfig output_config{
            dim,
            obs_val,
            sel.R_h_group,
            sel.V_vir_group,
            sel.R_h_iso,
            sel.V_vir_iso,
            config.tree_search,
            config.sat_reclass,
            config.iso_reclass,
            config.vel_cut,
            config.contrast,
            box_size,
            R_max,
            periodic,
            B_scaling,
            h_val
        };
        
        // Prepare statistics structure
        dataio::GroupFinderStatistics stats{
            total_groups,
            isolated_count,
            group_count
        };

        // Write results using HDF5 handler
        try {
            dataio::HDF5Handler::writeResults(outfilenames[ref], results, output_config, stats, datetime.str());
        } catch (const std::exception& e) {
            std::cerr << "Error writing results: " << e.what() << "\n";
            return 1;
        }
    }
    return 0;
}
