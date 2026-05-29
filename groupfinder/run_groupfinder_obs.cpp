#include "GroupFinderCore.hh"
#include "DataIO.hh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

bool stobool(const std::string& s) {
    std::istringstream iss(s);
    bool b;
    if (!(iss >> std::boolalpha >> b))
        throw std::invalid_argument("stobool: invalid input");
    return b;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_prefix> <config_file>\n";
        return 1;
    }
    const std::string infilename = argv[1];
    const std::string outfilename = argv[2];
    const std::string config_file = argv[3];

    using namespace gf;
    using json = nlohmann::json;

    // Get timestamp for output file
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");

    // Load observational data using HDF5 handler
    dataio::ObsInputData galaxy_data;
    try {
        galaxy_data = dataio::HDF5Handler::readObsData(infilename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading galaxy data: " << e.what() << "\n";
        return 1;
    }

    // Load concentration data
    dataio::ConcentrationData conc_data;
    try {
        conc_data = dataio::HDF5Handler::readConcentrationData("input/concentration.h5");
    } catch (const std::exception& e) {
        std::cerr << "Error loading concentration data: " << e.what() << "\n";
        return 1;
    }

    // Load redshift-distance data
    dataio::RedshiftDistanceData z_dist_data;
    try {
        z_dist_data = dataio::HDF5Handler::readRedshiftDistanceData("input/redshift_distance.h5");
    } catch (const std::exception& e) {
        std::cerr << "Error loading redshift-distances data: " << e.what() << "\n";
        return 1;
    }

    // Configure run parameters from config JSON file
    std::ifstream config_stream(config_file);
    if (!config_stream) {
        std::cerr << "Cannot open config file: " << config_file << "\n";
        return 1;
    }
    json config_json;
    config_stream >> config_json;

    int dim = config_json["dim"].get<int>(); // Should be 3 for observational run
    bool obs_val = config_json["obs"].get<bool>(); // Observational run, should be true
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
    double box_size = R_max; // For obs, box size can be set to any value
    // but note the code will select all galaxies within R_max if R_max >= box_size * sqrt(3)/2,
    // which could slow down the run and increase memory usage for very large datasets (O(10^5))
    bool periodic = config_json["periodic"].get<bool>();
    double B_scaling = config_json["B_scaling"].get<double>();
    double h_val = config_json["h"].get<double>();
    bool use_distance = config_json["use_distance"].get<bool>();
    double omega_M = config_json["omega_M"].get<double>();
    
    SelectionCriteria sel{
        R_h_group_val,
        V_vir_group_val,
        R_h_iso_val,
        V_vir_iso_val
    };
    
    GroupFinderSettings config{
        3,
        obs_val,
        vel_cut_val,
        tree_search_val,
        sat_reclass_val,
        iso_reclass_val,
        contrast_val,
        use_distance
    };

    std::vector<std::vector<IDType>> group_indices;
    std::vector<IDType> central_ids;
    std::vector<double> halo_masses;

    // Run group finder
    if (use_distance) {
        GroupFinder<DistObs, VelObs> finder(sel, config, box_size, h_val*100., omega_M, periodic);
        finder.set_conc_table(conc_data.halo_masses, conc_data.concentration, conc_data.redshifts);
        finder.set_z_dist_table(z_dist_data.redshifts, z_dist_data.distances);

        auto result = finder.run_once_obs(
            galaxy_data.masses, 
            galaxy_data.ids, 
            galaxy_data.positions, 
            galaxy_data.velocities, 
            R_max, 
            B_scaling, 
            periodic
        );

        group_indices = std::get<0>(result);
        central_ids = std::get<1>(result);
        halo_masses = std::get<2>(result);

    } else { // the zeroth column in positions is assumed to be redshifts
        GroupFinder<DistObs, VelTotal> finder(sel, config, box_size, h_val*100., omega_M, periodic);
        finder.set_conc_table(conc_data.halo_masses, conc_data.concentration, conc_data.redshifts);
        finder.set_z_dist_table(z_dist_data.redshifts, z_dist_data.distances);

        auto result = finder.run_once_obs(
            galaxy_data.masses, 
            galaxy_data.ids, 
            galaxy_data.positions, 
            galaxy_data.velocities, 
            R_max, 
            B_scaling, 
            periodic
        );

        group_indices = std::get<0>(result);
        central_ids = std::get<1>(result);
        halo_masses = std::get<2>(result);
    }

    // Compute group statistics
    int total_groups = 0;
    int isolated_count = 0;
    int group_count = 0;

    for (const auto& group : group_indices) {
        total_groups++;
        if (group.size() == 1) {
            isolated_count++;
        } else {
            group_count += group.size();
        }
    }
    
    std::cout << "Total groups = " << total_groups
              << ", Isolated centrals = " << isolated_count
              << ", Group members = " << group_count << std::endl;

    dataio::GroupFinderResults results;
    results.group_member_ids = group_indices;
    results.central_ids = central_ids;
    results.halo_masses = halo_masses;
    
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
        config.use_distance,
        box_size,
        R_max,
        periodic,
        B_scaling,
        h_val,
        omega_M
    };
    
    dataio::GroupFinderStatistics stats{
        total_groups,
        isolated_count,
        group_count
    };

    // Write results
    try {
        dataio::HDF5Handler::writeResults(outfilename, results, output_config, stats, datetime.str());
    } catch (const std::exception& e) {
        std::cerr << "Error writing results: " << e.what() << "\n";
        return 1;
    }

    return 0;
}