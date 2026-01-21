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
    double M0=12.035, alpha=1.963, beta=0.482, 
           delta=0.411, gamma=std::pow(10,-1.034), epsilon=-1.435;
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
inline constexpr double GF_P_CRIT = (3. * GF_H*GF_H) / (8. * M_PI * GF_G); // M_sun/Mpc^3

HaloProps compute_halo_props(
    double logMstar, const BehrooziParams& params = BehrooziParams(), double p_crit = GF_P_CRIT, double G = GF_G
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
    double operator()(const Vec3& mw_c, const Vec3& mw_s,
                      const Vec3& v_c, const Vec3& v_s, double H = GF_H) const;
};
struct VelObs {
    double operator()(const double& v_c, const double& v_s) const;
};

struct TransformOutput {
    std::vector<double> rel_dists; // relative distances, projected or perpendicular
    std::vector<double> rel_vels;   // relative LOS velocity
    std::vector<double> R_cen;    // distance to central
};

struct GroupFinderSettings {
    bool obs; // True if observational data, false if simulation data
    bool vel_cut; // True or false
    bool tree_search; // True or false
    bool sat_reclass; // True or false
    bool iso_reclass; // True or false
    bool contrast; // True or false
};

class AboriaNeighborBuilder; // Forward declare the kd tree builder class

template<class DistMethod, class VelMethod>
class GroupFinder {
public:
    GroupFinder(SelectionCriteria s, GroupFinderSettings cfg, double box_size = GF_BOX_SIZE, 
                double H0 = GF_H, bool pbc = true)
                : sel(s), config(cfg), L(box_size), H(H0), periodic(pbc) {}

    ~GroupFinder();

    void set_conc_table(std::vector<double> M_arr, std::vector<double> c_array);
    
    double c_from_M(double M) const;

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
    double L, H;
    bool periodic;
    std::vector<IDType> groupcat_ids_sorted; // sorted groupcat ids
    std::vector<double> masses_sorted;       // sorted masses in log_10 solar masses
    std::vector<Vec3> positions_sorted; // sorted 3D cartesian/spherical positions
    std::vector<Vec3> velocities_sorted; // sorted 3D velocities
    std::vector<double> velocities_sorted_obs;
    std::vector<Vec3> MWcoords;      // sorted minimal-image of each position relative to the MW position
    std::vector<int> mass_order;    // sorted indices by descending mass
    std::vector<IDType> group_label;   // local group labels (central local index)
    std::vector<int> classification; // 0 = group central, 1 = isolated central, 2 = satellite

    std::vector<IDType> local_ids; // local indices [0, N-1] into sorted arrays

    DistMethod dist_method;     // desired method for computing relative distances
    VelMethod vel_method;       // desired method for computing relative velocities

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

    std::vector<double> tab_logM_c_;
    std::vector<double> tab_logc_;
};
}

#include "kdtreeNeighborSearch.hh"

namespace gf {
    template<class D,class V>
    GroupFinder<D,V>::~GroupFinder() = default;
};