#pragma once
#include <vector>
#include <array>
#include <functional>
#include <limits>
#include <cmath>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <tuple>
#include <algorithm>

namespace gf {

using Vec3 = std::array<double,3>;
using IDType = std::int64_t;

struct BehrooziParams {
    double EPS_0 = -1.435,
    EPS_A = 1.831,
    EPS_LOGA = 1.368,
    EPS_Z = -0.217,
    M_0 = 12.035,
    M_A = 4.556,
    M_LOGA = 4.417,
    M_Z = -0.731,
    ALPHA_0 = 1.963,
    ALPHA_A = -2.316,
    ALPHA_LOGA = -1.732,
    ALPHA_Z = 0.178,
    BETA_0 = 0.482,
    BETA_A = -0.841,
    BETA_Z = -0.471,
    DELTA_0 = 0.411,
    GAMMA_0 = -1.034,
    GAMMA_A = -3.100,
    GAMMA_Z = -1.055;
};

struct SelectionCriteria {
    double R_h_group;
    double V_vir_group;
    double R_h_iso;
    double V_vir_iso;
};

struct HaloProps {
    double M_h{};
    double R_h{};
    double V_vir{};
};

inline constexpr double GF_h = 0.6774; // Hubble parameter h
inline constexpr double GF_H = GF_h*100.; // km/s/Mpc, can be defined from config
inline constexpr double GF_G = 4.30091727e-9; // Mpc (km/s)^2 / Msun
inline constexpr double GF_BOX_SIZE = 35000./GF_h/1000.; // Mpc, can be defined from config
inline constexpr double GF_C = 299792.458; // km/s
inline constexpr double GF_OMEGA_M = 0.3089;

HaloProps compute_halo_props(
    double z, double logMstar, double p_crit_0, 
    const BehrooziParams& params = BehrooziParams(), 
    double omega_m = GF_OMEGA_M
);

/* ################# Define distance methods ################ */
struct DistPerp2D {
    double operator()(const Vec3& r_cen_mw, const Vec3& r_sat_mw, double L = GF_BOX_SIZE) const;
};
struct Dist3D {
    double operator()(const Vec3& r_cen_mw, const Vec3& r_sat_mw, double L = GF_BOX_SIZE) const;
};
struct DistProjectedRCTan {
    double operator()(const Vec3& r_cen_mw, const Vec3& r_sat_mw, double L = GF_BOX_SIZE) const;
};
struct DistObs {
    double operator()(const double& RA_cen, const double& Dec_cen,
                      const double& RA_sat, const double& Dec_sat,
                      const double& R_cen) const;
};


/* ################# Define velocity methods ################ */
struct VelPeculiar3D {
    double operator()(const Vec3& v_c, const Vec3& v_s, double L = GF_BOX_SIZE) const;
};
struct VelPeculiar2D {
    double operator()(const Vec3& mw_c, const Vec3& mw_s,
                      const Vec3& v_c, const Vec3& v_s) const;
};
struct VelTotal {
    double operator()(const double& z_c, const double& z_s) const;
};
struct VelObs {
    double operator()(const double& v_c, const double& v_s) const;
};

struct TransformOutput {
    std::vector<double> rel_dists; // relative distances, projected or perpendicular
    std::vector<double> rel_vels;  // relative LOS velocity
    std::vector<double> R_cen;  // distance to central
};

struct GroupFinderSettings {
    int dim; // 3 or 6 for simulation data, 3 by default for observational data
    bool obs; // True if observational data, false if simulation data
    bool vel_cut; // True or false
    bool tree_search; // True or false
    bool sat_reclass; // True or false
    bool iso_reclass; // True or false
    bool contrast; // True or false
    bool use_distance; // True if using distance and peculiar velocity for obs classification, False if using redshift and velocity for obs classification
                        // This is generally always true for simulation data, since redshifts can be computed from distances and velocities in this code
};

class AboriaNeighborBuilder; // Forward declare the kd tree builder class

template<class DistMethod, class VelMethod>
class GroupFinder {
public:
    GroupFinder(SelectionCriteria s, GroupFinderSettings cfg, double box_size = GF_BOX_SIZE, 
                double H0 = GF_H, double OMEGA_M0 = GF_OMEGA_M, bool pbc = true)
                : sel(s), config(cfg), L(box_size), H(H0), OMEGA_M(OMEGA_M0), periodic(pbc) {}

    ~GroupFinder();

    void set_conc_table(std::vector<double> M_arr, std::vector<std::vector<double>> c_array, std::vector<double> redshift);
    
    double c_from_M(double M, double z) const;

    void set_z_dist_table(std::vector<double> z_arr, std::vector<double> D_arr);

    double z_from_D(double d) const;

    double D_from_z(double z) const;

    std::tuple<std::vector<std::vector<IDType>>, std::vector<IDType>, std::vector<double>>
    run_once(
        const std::vector<double>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_box,
        const std::vector<Vec3>& velocities_pec,
        const Vec3& MW_pos_box, const Vec3& MW_vel_pec,
        const double& Rmax, const double& scale,
        bool periodic = true);

    std::tuple<std::vector<std::vector<IDType>>, std::vector<IDType>, std::vector<double>>
    run_once_obs(
        const std::vector<double>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_unsorted,
        const std::vector<double>& velocities_los,
        const double& Rmax, const double& scale,
        bool periodic = false);

    SelectionCriteria sel;  // selection criteria for groups and isolated centrals
    GroupFinderSettings config;
private:
    double L, H, OMEGA_M;
    bool periodic;
    std::vector<IDType> groupcat_ids_sorted; // sorted groupcat ids
    std::vector<double> masses_sorted;  // sorted masses in log_10 solar masses
    std::vector<Vec3> positions_sorted; // sorted 3D cartesian/spherical positions
    std::vector<Vec3> velocities_sorted; // sorted 3D velocities
    std::vector<double> velocities_sorted_obs;
    std::vector<Vec3> MWcoords;  // sorted minimal-image of each position relative to the MW position
    std::vector<int> mass_order;  // sorted indices by descending mass
    std::vector<IDType> group_label; // local group labels (central local index)
    std::vector<int> classification; // 0 = group central, 1 = isolated central, 2 = satellite
    std::vector<double> total_redshifts;

    std::vector<IDType> local_ids; // local indices [0, N-1] into sorted arrays

    DistMethod dist_method; // desired method for computing relative distances
    VelMethod vel_method;  // desired method for computing relative velocities

    // function for group classification
    std::tuple<std::vector<std::vector<IDType>>, std::vector<IDType>, std::vector<double>>
    classify(const double& Rmax, const double& scale, const bool& periodic);

    // general function for transforming the 3D vectors to projected components
    TransformOutput transform(size_t central_local, const std::vector<IDType>& local_indices) const;
    
    void initialize(const std::vector<double>& masses_unsorted,
            const std::vector<IDType>& groupcat_ids,
            const std::vector<Vec3>& positions_box,
            const std::vector<Vec3>& velocities_pec,
            const Vec3& MW_pos_box, const Vec3& MW_vel_pec,
            bool periodic);

    void initialize_obs(const std::vector<double>& masses_unsorted,
            const std::vector<IDType>& groupcat_ids,
            const std::vector<Vec3>& positions_unsorted,
            const std::vector<double>& velocities_los);
    
    std::unique_ptr<AboriaNeighborBuilder> tree;
    std::vector<HaloProps> halo_props;

    void reassign_satellites(double Rmax, bool periodic, const double& scale);

    void reassign_isolated(double Rmax, bool periodic, const double& scale);

    std::array<double,2> density_contrast(int local_c_id, double trans_dist, double rel_vel);

    std::vector<double> tab_logM_c_; // log mass
    std::vector<std::vector<double>> tab_logc_;  // log concentration
    std::vector<double> tab_z_c_; // redshifts

    std::vector<double> tab_z_D_; // redshifts
    std::vector<double> tab_D_; // comoving distances
};
}

#include "kdtreeNeighborSearch.hh"

namespace gf {
    template<class D,class V>
    GroupFinder<D,V>::~GroupFinder() = default;
};