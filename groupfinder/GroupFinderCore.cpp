#include "GroupFinderCore.hh"
#include "kdtreeNeighborSearch.hh"
#include "BruteNeighborSearch.hh"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <limits>
#include <memory>
#include <type_traits>
#include <tuple>
#include <array>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <string>

namespace gf {

static inline double wrap(double x, double L) {
    // Wraps a value x to be within the bounds [-L/2, L/2]
    // x may be an absolute or a relative distance
    return x - L * std::round(x / L);
}

static inline Vec3 minimal(const Vec3& a,const Vec3& b,double L) {
    // Compute the minimal image distance vector between two points in a periodic box
    return {
        wrap(a[0]-b[0],L),wrap(a[1]-b[1],L),wrap(a[2]-b[2],L)
    };
}

static double behroozi_SMHM(const double& x, double current_mass, const BehrooziParams& params) {
    return params.M0 + params.epsilon - std::log10(std::pow(10, -params.alpha * x) + std::pow(10, -params.beta * x))
            + params.gamma * std::exp(-0.5 * std::pow(x / params.delta, 2)) - current_mass;
}

static inline double interp_lin(double x,
                                const std::vector<double>& x_arr,
                                const std::vector<double>& y_arr)
{
    // x_arr must be sorted in ascending order
    assert(x_arr.size() == y_arr.size() && x_arr.size() >= 2);
    if (x <= x_arr.front()) return y_arr.front();
    if (x >= x_arr.back()) return y_arr.back();
    auto it = std::lower_bound(x_arr.begin(), x_arr.end(), x); // first x_arr[j] >= x
    size_t j = size_t(it - x_arr.begin()); // get location of index just above x
    size_t i = j - 1; // index just below x
    // Linear interpolation
    double t = (x - x_arr[i]) / (x_arr[j] - x_arr[i]);
    return (1.0 - t) * y_arr[i] + t * y_arr[j];
}

// Secant root-finding method for inverting Behroozi SMHM relation
static double find_root(double x0, double x1, double current_mass, const BehrooziParams& params, double tolerance = 1e-9, int max_iter = 100) {
    double f_x0 = behroozi_SMHM(x0, current_mass, params);
    double f_x1 = behroozi_SMHM(x1, current_mass, params);

    for (int i = 0; i < max_iter; ++i) {
        if (std::fabs(f_x1) < tolerance) {
            return x1;
        }

        if (std::fabs(x1 - x0) < tolerance) {
            std::cout << "x0 and x1 are too close together. Iteration stopped.\n";
            return x1;
        }

        double denominator = (f_x1 - f_x0);

        if (std::fabs(denominator) < 1e-12) {
            std::cout << "Division by zero risk. Stopping iteration.\n";
            return x1;
        }

        double x2 = x1 - f_x1 * (x1 - x0) / denominator;

        x0 = x1;
        f_x0 = f_x1;
        x1 = x2;
        f_x1 = behroozi_SMHM(x1, current_mass, params);
    }

    std::cout << "Max iterations reached without convergence. Last approximation: " << x1 << "\n";
    return x1;
}

// Precompute halo properties once per iteration of group_finding
HaloProps compute_halo_props(double logMstar, const BehrooziParams& params, double p_crit, double G) {
    double sol = find_root(-5., 4., logMstar, params);
    double M_h = std::pow(10., sol + params.M0); // Msun
    double R_h = std::cbrt((3.*M_h)/(4.*M_PI*200.*p_crit)); // Mpc
    double V_vir = std::sqrt(G * M_h / R_h); // km/s
    return {M_h, R_h, V_vir};
}

void summarize(const std::string& label, const std::vector<IDType>& group_label){
    std::unordered_map<IDType,int> group_sizes;
    group_sizes.reserve(group_label.size());
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        if (g == -1) {
            std::cerr << "Warning: galaxy " << i << " is unassigned to any group.\n";
            std::abort();
        } else { // Count the number of members in each group (all have same group label as central)
            ++group_sizes[g];
        }
    }
    size_t n_groups = group_sizes.size();
    size_t n_single = 0;
    for (auto &kv: group_sizes) if (kv.second==1) ++n_single;
    std::cout << label << ": groups=" << n_groups
              << " single=" << n_single
              << " multi=" << (n_groups - n_single) << "\n";
};

/* ################# Define distance methods ################ */

double Dist3D::operator()(const Vec3& mw_c, const Vec3& mw_s, double L) const {
    // Compute the relative distance between the satellite and the central
    Vec3 d = minimal(mw_s, mw_c, L);
    double dist2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    double dist = 0.;
    if (dist2 > 0.) dist = std::sqrt(dist2);
    return dist;
};

/* The below method is used for the 2D distance implementation. */
double DistPerp2D::operator()(const Vec3& mw_c, const Vec3& mw_s, double L) const {

    // Compute the relative distance between the satellite and the central
    Vec3 d = minimal(mw_s, mw_c, L);
    
    double rc2 = mw_c[0]*mw_c[0]+mw_c[1]*mw_c[1]+mw_c[2]*mw_c[2];
    if(rc2==0.) return 0.0; // This case occurs if the central is the MW analogue

    // Compute the normal vector for the line-of-sight distance to the central
    double rc = std::sqrt(rc2);
    Vec3 n = {mw_c[0]/rc, mw_c[1]/rc, mw_c[2]/rc};

    // Take the dot product of the distance vector and the line-of-sight central normal vector
    double dpar = d[0]*n[0] + d[1]*n[1] + d[2]*n[2];

    // Compute the squared perpendicular distance
    double perp2 = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]) - dpar*dpar;
    return perp2 <=0 ? 0.0 : std::sqrt(perp2);
}

/* The below method is used for the 2D distance implementation that mimics the observational group finder */
double DistProjectedRCTan::operator()(const Vec3& r_cen_mw, const Vec3& r_sat_mw, double L) const {

    // Compute the squared distance of the central from the MW
    double rc2 = r_cen_mw[0]*r_cen_mw[0] + r_cen_mw[1]*r_cen_mw[1] + r_cen_mw[2]*r_cen_mw[2];
    if(rc2==0.) return 0.0; // This case occurs if the central is the MW
    double rc_len = std::sqrt(rc2);

    // Compute the relative distance between the satellite and the central with PBCs applied
    Vec3 dcs = minimal(r_sat_mw,r_cen_mw,L);

    // Compute the coordinate positions of the satellite in the wrapped coordinates
    Vec3 rs = {r_cen_mw[0]+dcs[0], r_cen_mw[1]+dcs[1], r_cen_mw[2]+dcs[2]};
    double rs2 = rs[0]*rs[0] + rs[1]*rs[1] + rs[2]*rs[2];

    if(rs2==0.) return 0.0; // This case is unlikely to occur but avoids division by zero
    double rs_len = std::sqrt(rs2);
    double cos_theta = (r_cen_mw[0]*rs[0] + r_cen_mw[1]*rs[1] + r_cen_mw[2]*rs[2])/(rc_len*rs_len);

    // Clamp numerically
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    // Only forward intersection allowed
    if (cos_theta <= 0.0) return std::numeric_limits<double>::infinity();

    // Compute the haversine distance
    double h = 0.5*(1.0 - cos_theta);

    // Use the small-angle approximation if h is too small
    if (h < 1e-14) return rc_len * (2.0*std::sqrt(h));

    // Otherwise, compute tan_theta in a numerically stable way
    double sin_theta = 2.0*std::sqrt(h*(1.0 - h));
    double tan_theta = sin_theta / cos_theta;
    return rc_len * tan_theta;
}

double DistObs::operator()(const double& RA_cen, const double& Dec_cen,
                      const double& RA_sat, const double& Dec_sat,
                      const double& R_cen) const {
    double dRA = RA_sat - RA_cen;
    double dDec = Dec_sat - Dec_cen;

    // Wrap dRA into the appropriate bounds
    if (dRA > M_PI) dRA -= 2.0*M_PI;
    if (dRA < -M_PI) dRA += 2.0*M_PI;

    // Obtain the haversine of theta
    double h = std::sin(0.5*dDec)*std::sin(0.5*dDec) + std::cos(Dec_cen)*std::cos(Dec_sat)*std::sin(0.5*dRA)*std::sin(0.5*dRA);

    // Clamp to [0, 1]
    h = std::min(1.0, std::max(0.0, h));
    double cos_theta = 1.0-2.0*h;

    // No forward intersection if cos_theta <= 0
    if (cos_theta <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    // For very small angles, use the small angle approximation
    if (h < 1e-14) {
        return R_cen * 2.0 * std::sqrt(h);
    }

    double sin_theta = 2.0*std::sqrt(h*(1.0-h));

    // R_proj = R0 * tan(theta)
    return R_cen * (sin_theta / cos_theta);
}

/* ################# Define velocity methods ################ */
/* The below method mimics velocity criteria implementation of an observational group finder */
double VelTotal::operator()(const Vec3& mw_c, const Vec3& mw_s,
                      const Vec3& v_c, const Vec3& v_s, double H) const {
                        
    double rc2 = mw_c[0]*mw_c[0] + mw_c[1]*mw_c[1] + mw_c[2]*mw_c[2];
    double rs2 = mw_s[0]*mw_s[0] + mw_s[1]*mw_s[1] + mw_s[2]*mw_s[2];
    if(rc2==0.) return 0.0; // This case occurs if the central is the MW
    if(rs2==0.) return 0.0; // This case occurs if the satellite is the MW

    double rc = std::sqrt(rc2);
    double rs = std::sqrt(rs2);

    // Compute the normal vector for the line-of-sight distance to the central
    Vec3 nc = {mw_c[0]/rc, mw_c[1]/rc, mw_c[2]/rc};
    // ... and to the satellite
    Vec3 ns = {mw_s[0]/rs, mw_s[1]/rs, mw_s[2]/rs};

    // Compute the velocity of the central projected along its line-of-sight normal vector
    double vc_LOS = (v_c[0] * nc[0] + v_c[1] * nc[1] + v_c[2] * nc[2]);
    // ... and do the same for the satellite
    double vs_LOS = (v_s[0] * ns[0] + v_s[1] * ns[1] + v_s[2] * ns[2]);

    // Now compute and add the Hubble velocity
    double hubble_c = H * rc;
    double hubble_s = H * rs;

    // Return the relative velocity
    return (vs_LOS + hubble_s) - (vc_LOS + hubble_c);
}

/* The below methods are relevant for implementations considering peculiar velocity only */
double VelPeculiar3D::operator()(const Vec3& v_c, const Vec3& v_s, double L) const {
    Vec3 rel_vel = {v_s[0]-v_c[0], v_s[1]-v_c[1], v_s[2]-v_c[2]};
    return std::sqrt(rel_vel[0]*rel_vel[0] + rel_vel[1]*rel_vel[1] + rel_vel[2]*rel_vel[2]);
}

double VelPeculiar2D::operator()(const Vec3& mw_c, const Vec3& mw_s,
                                const Vec3& v_c, const Vec3& v_s) const {
    
    double rc2 = mw_c[0]*mw_c[0] + mw_c[1]*mw_c[1] + mw_c[2]*mw_c[2];
    double rs2 = mw_s[0]*mw_s[0] + mw_s[1]*mw_s[1] + mw_s[2]*mw_s[2];
    if(rc2==0.) return 0.0; // This case occurs if the central is the MW
    if(rs2==0.) return 0.0; // This case occurs if the satellite is the MW

    double rc = std::sqrt(rc2);
    double rs = std::sqrt(rs2);

    // Compute the normal vector for the line-of-sight distance to the central
    Vec3 nc = {mw_c[0]/rc, mw_c[1]/rc, mw_c[2]/rc};
    // ... and to the satellite
    Vec3 ns = {mw_s[0]/rs, mw_s[1]/rs, mw_s[2]/rs};

    // Compute the velocity of the central projected along its line-of-sight normal vector
    double vc_LOS = (v_c[0] * nc[0] + v_c[1] * nc[1] + v_c[2] * nc[2]);
    // ... and do the same for the satellite
    double vs_LOS = (v_s[0] * ns[0] + v_s[1] * ns[1] + v_s[2] * ns[2]);
    return vs_LOS - vc_LOS;
}

double VelObs::operator()(const double& v_c, const double& v_s) const {
    return std::abs(v_s - v_c);
}

template<class D,class V>
double GroupFinder<D,V>::c_from_M(double logM) const {
    if (tab_logM_c_.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double logc = interp_lin(logM, tab_logM_c_, tab_logc_); // x in log10(M), y in log10(c)
    return std::pow(10.0, logc);
}

template<class D,class V>
void GroupFinder<D,V>::set_conc_table(std::vector<double> M, std::vector<double> c_array) {
    const size_t N = std::min(M.size(), c_array.size());
    if (N < 2) {
        std::cerr << "Warning: concentration table must have at least two entries.\n";
        std::abort();
    }
    // Copy and build logs (assumes monotonic/unique already)
    tab_logM_c_.resize(N);
    tab_logc_.resize(N);
    for (size_t i = 0; i < N; ++i) {
        tab_logc_[i] = std::log10(c_array[i]);
    }
    tab_logM_c_ = M;
}

// This will generally take the index of one galaxy and the indices of its candidate neighbors
template<class D,class V>
TransformOutput GroupFinder<D,V>::transform(size_t central_local, const std::vector<IDType>& local_indices) const {
    TransformOutput tr;
    tr.rel_dists.resize(local_indices.size());
    tr.rel_vels.resize(local_indices.size());
    tr.R_cen.resize(local_indices.size());
    if (!config.obs) { // Check that this is simulation data
        const auto& mw_c = MWcoords[central_local];
        const auto& v_cen  = velocities_sorted[central_local];
        // Compute the distances and relative velocities
        for(size_t i = 0; i < local_indices.size(); ++i){
            size_t local_id = local_indices[i];
            const auto& mw_s = MWcoords[local_id];
            const auto& v_sat  = velocities_sorted[local_id];
            if constexpr (std::is_same_v<D,DistObs>) {
                std::cerr << "Error: Distances must not be of type DistObs when using the 3D/2D distance methods.\n";
                std::abort();
            } else {
                tr.rel_dists[i] = dist_method(mw_c,mw_s,L);
            }
            if constexpr (std::is_same_v<V,VelPeculiar3D>) {
                tr.rel_vels[i] = vel_method(v_cen,v_sat,L);
            } else if constexpr (std::is_same_v<V,VelPeculiar2D>) {
                tr.rel_vels[i] = vel_method(mw_c,mw_s,v_cen,v_sat);
            } else if constexpr (std::is_same_v<V,VelTotal>) {
                tr.rel_vels[i] = vel_method(mw_c,mw_s,v_cen,v_sat,H);
            }
        } 
    } else { // Observational data
        const auto& R_cen = positions_sorted[central_local][0];
        const auto& Dec_cen = positions_sorted[central_local][1];
        const auto& RA_cen = positions_sorted[central_local][2];
        const auto& v_cen  = velocities_sorted_obs[central_local];
        for (size_t i = 0; i < local_indices.size(); ++i) {
            size_t local_id = local_indices[i];
            const auto& Dec_sat = positions_sorted[local_id][1];
            const auto& RA_sat = positions_sorted[local_id][2];
            if constexpr (std::is_same_v<V,VelObs>) {
                tr.rel_vels[i] = vel_method(v_cen, velocities_sorted_obs[local_id]);
            } else {
                std::cerr << "Error: Velocities must be of type VelObs when using the observational distance method.\n";
                std::abort();
            }
            if constexpr (std::is_same_v<D,DistObs>) {
                tr.rel_dists[i] = dist_method(RA_cen, Dec_cen, RA_sat, Dec_sat, R_cen);
                tr.R_cen[i] = R_cen;
            } else {
                std::cerr << "Error: Distances must be of type DistObs when using the observational distance method.\n";
                std::abort();
            }
        }
    }
    return tr;
}

template<class D,class V>
std::array<double, 2> GroupFinder<D,V>::density_contrast(int local_c_id, double trans_dist, double rel_vel) {
    // Get V_vir, M200 and r200 from halo props
    double V_vir = halo_props[local_c_id].V_vir;
    double M_h = halo_props[local_c_id].M_h;
    double R_200 = halo_props[local_c_id].R_h;

    // Compute concentration from halo mass
    double c = c_from_M(std::log10(M_h));

    double r_s = R_200/c;
    double sigma_v = V_vir/std::sqrt(2.); // units of km/s
    double x = trans_dist / r_s;

    auto p = [&](double rel_vel) -> double {
        return (1./(std::sqrt(2.*M_PI)*sigma_v)) * std::exp(-std::pow(rel_vel, 2.) / (2.*sigma_v*sigma_v));
    }; // units of (km/s)^-1
    
    auto f = [&](double x) -> double { // dimensionless
        double val = 0.;
        if (x == 1.) {
            val = 1./3.;
        } else if (x < 1.) {
            val = (1./(x*x - 1.))*(1. - (std::log((1. + std::sqrt(1. - x*x))/x) / std::sqrt(1. - x*x)));
        } else if (x > 1.) {
            val = (1./(x*x - 1.))*(1. - (std::atan(std::sqrt(x*x - 1.)) / std::sqrt(x*x - 1.)));
        }
        return val;
    };

    double delta = (200./3.)*((c*c*c) / (std::log(1.+c) - c/(1.+c))); // dimensionless
    double Sigma_R_rho_gal = 2. * r_s * delta * f(x); // Units of Mpc

    // estimate projected surface density at R200
    double rho_200_rho_crit = delta / (c * (1. + c)*(1. + c)); // dimensionless
    double B = rho_200_rho_crit * (4.*R_200*gf::GF_H)/(3.*sigma_v); // dimensionless
    double P_M = gf::GF_H * Sigma_R_rho_gal * p(rel_vel);
    return {P_M, B};
}

template<class D,class V>
void GroupFinder<D,V>::reassign_satellites(double Rmax, bool periodic, const double& scale) {
    // Implementation for reassigning satellites
    // Build a tree of all group centrals and isolated centrals
    assert(classification.size() == positions_sorted.size());
    std::vector<IDType> central_indices; // indices into positions_sorted
    std::vector<Vec3> central_positions;
    std::vector<IDType> satellite_indices; // indices into positions_sorted
    std::unordered_map<IDType, int> central_map; // map from index in positions_sorted to index in central_indices
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0 || classification[i] == 1) { // 0 = group central, 1 = isolated central
            central_indices.push_back((IDType)i);
            central_positions.push_back(positions_sorted[i]);
            central_map[(IDType)i] = (int)central_indices.size() - 1;
        } else if (classification[i] == 2) {
            satellite_indices.push_back((IDType)i);
        } else {
            std::cerr << "Warning: Galaxy without a classification found.\n";
            std::abort();
        }
    }
    if (config.tree_search) {
        tree = std::make_unique<AboriaNeighborBuilder>(central_positions, central_indices, L, periodic);
    }
    int reclassified = 0;
    size_t N = satellite_indices.size();
    std::vector<double> B_values(central_indices.size());
    for (size_t s = 0; s < N; ++s) {
        IDType local_s_id = satellite_indices[s]; // index into positions_sorted
        std::vector<IDType> cand;
        if (config.tree_search) { // Returns local indices
            cand = tree->kdtree_search((size_t)local_s_id, positions_sorted, Rmax); // returns value from central_indices
        } else { // Returns local indices
            if (Rmax >= GF_BOX_SIZE * std::sqrt(3) / 2.0) {
                cand = central_indices; // All centrals are candidates
            } else {
                cand = bruteforce_search((size_t)local_s_id, positions_sorted, central_indices, Rmax, L, periodic, config.obs); // returns value from central_indices
            }
        }
        if (cand.empty()) {
            std::cerr << "Warning: No candidate centrals found for satellite index " << local_s_id << "\n";
            std::abort();
        }

        std::vector<double> ratios;
        std::vector<double> P_M_cands;
        std::vector<IDType> new_cands;
        for (size_t k = 0; k < cand.size(); ++k) {
            size_t local_c_id = cand[k]; // index into positions_sorted
            auto tr = transform((size_t)local_c_id, (std::vector<IDType>){local_s_id});
            double rel_dist = tr.rel_dists[0];
            double rel_vel = tr.rel_vels[0];
            double R_h_cand = halo_props[local_c_id].R_h;
            double V_vir_cand = halo_props[local_c_id].V_vir;
            double ratio = std::pow(rel_dist / (sel.R_h_group * R_h_cand), 2.) + std::pow(rel_vel / (sel.V_vir_group * V_vir_cand / std::sqrt(2.0)), 2.);
            double mass_cand = masses_sorted[local_c_id];
            std::array<double,2> dens = density_contrast(local_c_id, rel_dist, rel_vel);
            double P_M = dens[0];
            double B = dens[1]*scale;
            if (config.contrast) {
                B_values[(size_t)central_map[local_c_id]] = B;
                if (P_M > B && masses_sorted[(size_t)local_s_id] < mass_cand) {
                    new_cands.push_back(cand[k]);
                    P_M_cands.push_back(P_M);
                }
            } else {
                if (config.vel_cut) {
                    if (rel_dist <= sel.R_h_group * R_h_cand && masses_sorted[(size_t)local_s_id] < mass_cand
                        && std::fabs(rel_vel) <= sel.V_vir_group * V_vir_cand / std::sqrt(2.0)) {
                        ratios.push_back(ratio);
                        new_cands.push_back(cand[k]);
                    }
                } else {
                    if (rel_dist <= sel.R_h_group * R_h_cand && masses_sorted[(size_t)local_s_id] < mass_cand) {
                        ratios.push_back(ratio);
                        new_cands.push_back(cand[k]);
                    }
                }
            }
        }

        if (config.contrast) {
            int max_index = -1;
            if (P_M_cands.empty()) {
                // The satellite remains in its current group
                // Unlikely to occur, since a satellite must meet all criteria above to have been assigned to a group in part 1
                std::cerr << "Warning: No candidate centrals passed criteria for satellite index " << local_s_id << ".\n";
                continue;
            } else if (P_M_cands.size() == 1) {
                max_index = 0;
                IDType local_c_id = new_cands[max_index];
                // If the assigned central is different from the one assigned in part 1, count it as reclassified
                if (group_label[(size_t)local_s_id] != local_c_id) {
                    reclassified += 1;
                    group_label[(size_t)local_s_id] = local_c_id;
                }
            } else {
                // More than one candidate remains, assign to group with largest value of P_M
                auto max_it = std::max_element(P_M_cands.begin(), P_M_cands.end());
                max_index = std::distance(P_M_cands.begin(), max_it);
                // Get the ID of the central assigned to the satellite
                IDType local_c_id = new_cands[max_index];
                // If the assigned central is different from the one assigned in part 1, count it as reclassified
                if (group_label[(size_t)local_s_id] != local_c_id) {
                    reclassified += 1;
                    group_label[(size_t)local_s_id] = local_c_id;
                }
            }
        } else {
            int min_index = -1;
            if (ratios.empty()) {
                // The satellite remains in its current group
                // Unlikely to occur, since a satellite must meet all criteria above to have been assigned to a group in part 1
                std::cerr << "Warning: No candidate centrals passed criteria for satellite index " << local_s_id << ".\n";
                continue;
            } else if (ratios.size() == 1) {
                min_index = 0;
                IDType local_c_id = new_cands[min_index];
                // If the assigned central is different from the one assigned in part 1, count it as reclassified
                if (group_label[(size_t)local_s_id] != local_c_id) {
                    reclassified += 1;
                    group_label[(size_t)local_s_id] = local_c_id;
                }
            } else {
                // More than one candidate remains, find the one with the smallest ratio
                auto min_it = std::min_element(ratios.begin(), ratios.end());
                min_index = std::distance(ratios.begin(), min_it);
                // Get the ID of the central assigned to the satellite
                IDType local_c_id = new_cands[min_index];
                // If the assigned central is different from the one assigned in part 1, count it as reclassified
                if (group_label[(size_t)local_s_id] != local_c_id) {
                    reclassified += 1;
                    group_label[(size_t)local_s_id] = local_c_id;
                }
            }
        }
        cand.clear();
    }

    std::unordered_map<IDType,int> group_sizes;
    group_sizes.reserve(group_label.size());
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        if (g == IDType(-1)) {
            std::cerr << "Warning: galaxy " << i << " is unassigned to any group.\n";
            std::abort();
        } else ++group_sizes[g];
    }
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        int sz = group_sizes[g];
        if (sz == 1) {
            if (classification[i] == 2) {
                std::cerr << "Warning: Satellite assigned to a group of size 1 found. This should not happen.\n";
                std::abort();
            } else {
                classification[i] = 1; // isolated central
                group_label[i] = (IDType)i; // ensure self-label
            }
        } else if (sz > 1) {
            if (classification[i] == 1) classification[i] = 0; // reclassify isolated central to group central
        } 
    }
    std::cout << reclassified << " satellites reclassified to different groups\n";
    summarize("After satellite reclassification", group_label);
}

template<class D,class V>
void GroupFinder<D,V>::reassign_isolated(double Rmax, bool periodic, const double& scale) {
    // Build tree of group centrals only
    std::vector<IDType> group_central_indices;
    std::vector<Vec3> group_central_positions;
    std::vector<IDType> isolated_central_indices;
    assert(classification.size() == positions_sorted.size());
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0) { // 0 = group central
            group_central_indices.push_back(i);
            group_central_positions.push_back(positions_sorted[i]);
        } else if (classification[i] == 1) { // 1 = isolated central
            isolated_central_indices.push_back(i);
        }
    }
    if (config.tree_search) {
        tree = std::make_unique<AboriaNeighborBuilder>(group_central_positions, group_central_indices, L, periodic);
    }

    size_t N = isolated_central_indices.size();
    int reclassified = 0;
    for (size_t i = 0; i < N; ++i) {
        IDType local_i_id = isolated_central_indices[i];
        std::vector<IDType> cand;
        if (config.tree_search) {
            cand = tree->kdtree_search((size_t)local_i_id, positions_sorted, Rmax); // value from group_central_indices
        } else {
            if (Rmax >= GF_BOX_SIZE * std::sqrt(3) / 2.0) {
                cand = group_central_indices; // All group centrals are candidates
            } else {
                cand = bruteforce_search((size_t)local_i_id, positions_sorted, group_central_indices, Rmax, L, periodic, config.obs); // value from group_central_indices
            }
        }
        if (cand.empty()) {
            std::cerr << "Warning: No candidate group centrals found for isolated central index " << local_i_id << "\n";
            std::abort();
        }
        std::vector<double> ratios(cand.size());
        std::vector<double> P_M_cands(cand.size());
        std::vector<double> B_cands(cand.size());
        std::vector<double> relative_velocities(cand.size());
        std::vector<double> relative_distances(cand.size());
        std::vector<double> halo_radii(cand.size());
        std::vector<double> halo_velocities(cand.size());
        for (size_t k = 0; k < cand.size(); ++k) {
            size_t local_gc_id = cand[k]; // index into positions_sorted
            auto tr = transform(local_gc_id, (std::vector<IDType>){local_i_id});
            relative_distances[k] = tr.rel_dists[0];
            relative_velocities[k] = tr.rel_vels[0];
            halo_radii[k] = halo_props[local_gc_id].R_h;
            halo_velocities[k] = halo_props[local_gc_id].V_vir;
            ratios[k] = std::pow(tr.rel_dists[0] / (sel.R_h_group * halo_props[local_gc_id].R_h), 2.) + std::pow(tr.rel_vels[0] / (sel.V_vir_group * halo_props[local_gc_id].V_vir / std::sqrt(2.0)), 2.);
            std::array<double,2> dens = density_contrast(local_gc_id, relative_distances[k], tr.rel_vels[0]);
            P_M_cands[k] = dens[0];
            B_cands[k] = dens[1]*scale;
        }
        // To remain an isolated central, it must be more than twice the virial radius away from any group central
        bool isolated = true;
        for (int k = 0; k < cand.size(); ++k) {
            if (relative_distances[k] <= sel.R_h_iso*halo_radii[k]
                && relative_velocities[k] <= sel.V_vir_iso*halo_velocities[k] / std::sqrt(2.0)) {
                isolated = false;
                break;
            }
        }
        if (config.contrast) {
            if (isolated == true) {
                // compute the background levels relative to the 10 nearest centrals
                std::vector<int> dist_order(cand.size());
                std::iota(dist_order.begin(), dist_order.end(), 0);
                std::stable_sort(dist_order.begin(), dist_order.end(),
                    [&](int a,int b){return relative_distances[a] < relative_distances[b];});
                std::vector<double> dists_sorted(cand.size());
                std::vector<double> B_sorted(cand.size());
                std::vector<double> vels_sorted(cand.size());
                std::vector<double> V_vir_sorted(cand.size());
                std::vector<double> R_h_sorted(cand.size());
                std::vector<double> P_M_sorted(cand.size());
                for (int m = 0; m < cand.size(); ++m) {
                    dists_sorted[m] = relative_distances[dist_order[m]];
                    B_sorted[m] = B_cands[dist_order[m]];
                    vels_sorted[m] = relative_velocities[dist_order[m]];
                    V_vir_sorted[m] = halo_velocities[dist_order[m]];
                    R_h_sorted[m] = halo_radii[dist_order[m]];
                    P_M_sorted[m] = P_M_cands[dist_order[m]];
                }
            } else if (isolated == false) {
                bool found_central = false;
                int max_index = -1;
                int N_exit = 0;
                // Copy the ratios vector to avoid modifying the original
                std::vector<double> P_M_copy = P_M_cands;
                while (!found_central && N_exit < cand.size()) {
                    auto max_it = std::max_element(P_M_copy.begin(), P_M_copy.end());
                    // Safety check: if maximum P_M is minimal, all candidates exhausted
                    if (*max_it == std::numeric_limits<double>::lowest()) break;
                    // Compute the index of the maximum P_M
                    max_index = std::distance(P_M_copy.begin(), max_it);
                    if (P_M_cands[max_index] > (sel.R_h_iso/sel.V_vir_iso) * B_cands[max_index]
                        && masses_sorted[(size_t)local_i_id] < masses_sorted[cand[max_index]]) {
                        found_central = true;
                    } else {
                        // If the relative velocity is too high, remove this candidate (set to lowest)
                        P_M_copy[max_index] = std::numeric_limits<double>::lowest();
                        continue;
                    }
                    N_exit++; // Increment exit counter to avoid infinite loop
                } 

                // If the isolated central does not meet the velocity criteria, simply keep it as a field galaxy
                if (!found_central) continue;
                else {
                    // Get the group ID of the central assigned to the isolated central and reassign
                    IDType local_c_id = cand[max_index];
                    if (local_c_id != local_i_id) {
                        classification[(size_t)local_i_id] = 2; // Reclassified as satellite
                        group_label[(size_t)local_i_id] = local_c_id; // Reassign to the new group
                        reclassified += 1;
                    }
                }
            }
        } else {
            if (isolated == false) {
                bool found_central = false;
                int min_index = -1;
                int N_exit = 0;
                // Copy the ratios vector to avoid modifying the original
                std::vector<double> ratios_copy = ratios;
                while (!found_central && N_exit < cand.size()) {
                    auto min_it = std::min_element(ratios_copy.begin(), ratios_copy.end());
                    // Safety check: if minimum ratio is infinity, all candidates exhausted
                    if (*min_it == std::numeric_limits<double>::max()) break;
                    // Compute the index of the minimum ratio
                    min_index = std::distance(ratios_copy.begin(), min_it);
                    if (config.vel_cut) {
                        if (std::fabs(relative_velocities[min_index]) <= sel.V_vir_iso*halo_velocities[min_index] / std::sqrt(2.0) 
                            && relative_distances[min_index] <= sel.R_h_iso*halo_radii[min_index]
                            && masses_sorted[(size_t)local_i_id] < masses_sorted[cand[min_index]]
                        ) {
                            found_central = true;
                        } else {
                            // If the relative velocity is too high, remove this candidate (set to max)
                            ratios_copy[min_index] = std::numeric_limits<double>::max();
                            continue;
                        }
                    } else {
                        if (relative_distances[min_index] <= sel.R_h_iso*halo_radii[min_index] 
                            && masses_sorted[(size_t)local_i_id] < masses_sorted[cand[min_index]]) {
                            found_central = true;
                        } else {
                            ratios_copy[min_index] = std::numeric_limits<double>::max();
                            continue;
                        }
                    }
                    N_exit++; // Increment exit counter to avoid infinite loop
                } 

                // If the isolated central does not meet the velocity criteria, simply keep it as a field galaxy
                if (!found_central) continue;
                else {
                    // Get the group ID of the central assigned to the isolated central and reassign
                    IDType local_c_id = cand[min_index];
                    if (local_c_id != local_i_id) {
                        classification[(size_t)local_i_id] = 2; // Reclassified as satellite
                        group_label[(size_t)local_i_id] = local_c_id; // Reassign to the new group
                        reclassified += 1;
                    }
                }
            }
        }
        cand.clear();
    }
    //outfile.close();
    std::cout << reclassified << " isolated centrals reclassified as satellites\n";
    summarize("After isolated central classification", group_label);
}

template<class D,class V>
void GroupFinder<D,V>::initialize(const std::vector<double>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_box,
        const std::vector<Vec3>& velocities_pec,
        const Vec3& MW_pos_box,
        const Vec3& MW_vel_pec,
        bool periodic) {

    // Sort the galaxies by descending mass
    size_t N = masses_unsorted.size();
    mass_order.resize(N);
    std::iota(mass_order.begin(), mass_order.end(), 0);
    std::stable_sort(mass_order.begin(), mass_order.end(),
              [&](int a,int b){return masses_unsorted[a] > masses_unsorted[b];});

    masses_sorted.resize(N);
    groupcat_ids_sorted.resize(N);
    positions_sorted.resize(N);
    velocities_sorted.resize(N);
    MWcoords.resize(N);
    classification.resize(N);
    halo_props.resize(N);
    local_ids.resize(N);

    for(size_t i = 0; i < N; ++i) {
        local_ids[i] = (IDType)i;
        positions_sorted[i]  = positions_box[mass_order[i]];
        velocities_sorted[i] = velocities_pec[mass_order[i]];
        masses_sorted[i] = masses_unsorted[mass_order[i]];
        // Compute the positions and velocities relative to the MW
        Vec3 d = {positions_sorted[i][0]-MW_pos_box[0],
                  positions_sorted[i][1]-MW_pos_box[1],
                  positions_sorted[i][2]-MW_pos_box[2]};
        Vec3 v = {velocities_sorted[i][0]-MW_vel_pec[0],
                  velocities_sorted[i][1]-MW_vel_pec[1],
                  velocities_sorted[i][2]-MW_vel_pec[2]};
        // By default, assume periodic boundary conditions
        if (periodic) {
            d[0] = wrap(d[0],L); d[1] = wrap(d[1],L); d[2] = wrap(d[2],L);
        }
        MWcoords[i] = d;
        velocities_sorted[i] = v;
        halo_props[i] = compute_halo_props(masses_unsorted[mass_order[i]], 
                                                  gf::BehrooziParams(),
                                                  gf::GF_P_CRIT,
                                                  gf::GF_G);
        groupcat_ids_sorted[i] = groupcat_ids[mass_order[i]];
    }
}

template<class D,class V>
void GroupFinder<D,V>::initialize_obs(const std::vector<double>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_unsorted,
        const std::vector<double>& velocities_los) {

    // Sort the galaxies by descending mass
    size_t N = masses_unsorted.size();
    mass_order.resize(N);
    std::iota(mass_order.begin(), mass_order.end(), 0);
    std::stable_sort(mass_order.begin(), mass_order.end(),
              [&](int a,int b){return masses_unsorted[a] > masses_unsorted[b];});

    masses_sorted.resize(N);
    groupcat_ids_sorted.resize(N);
    positions_sorted.resize(N);
    velocities_sorted_obs.resize(N);
    classification.resize(N);
    halo_props.resize(N);
    local_ids.resize(N);

    for(size_t i = 0; i < N; ++i) {
        local_ids[i] = (IDType)i;
        positions_sorted[i]  = positions_unsorted[mass_order[i]];
        velocities_sorted_obs[i] = velocities_los[mass_order[i]];
        masses_sorted[i] = masses_unsorted[mass_order[i]];
        halo_props[i] = compute_halo_props(masses_unsorted[mass_order[i]], 
                                                  gf::BehrooziParams(),
                                                  gf::GF_P_CRIT,
                                                  gf::GF_G);
        groupcat_ids_sorted[i] = groupcat_ids[mass_order[i]];
    }
}

template<class D,class V>
std::tuple<std::vector<std::vector<IDType>>, std::vector<IDType>, std::vector<double>> 
GroupFinder<D,V>::classify(const double& Rmax, const double& scale, const bool& periodic) {
    // If kdtree, initialize the AboriaNeighborBuilder class
    // Rebuild the tree upon every call to the class
    if (config.tree_search && !config.obs) {
        tree = std::make_unique<AboriaNeighborBuilder>(positions_sorted, local_ids, L, periodic);
    }
    size_t N = masses_sorted.size();
    group_label.assign(N, IDType(-1)); // -1 means unassigned

    for(size_t c = 0; c < N; ++c){
        if (group_label[c] != IDType(-1)) continue; // The central has already been assigned a group

        // Find the candidate satellites
        std::vector<IDType> cand;
        if (config.tree_search) {
            cand = tree->kdtree_search(c, positions_sorted, 3.0*halo_props[c].R_h);
        } else {
            if (Rmax >= GF_BOX_SIZE * std::sqrt(3) / 2.0) {
                cand = local_ids; // All galaxies are candidates
            } else {
                cand = bruteforce_search(c, positions_sorted, local_ids, Rmax, L, periodic, config.obs);
            }
        }
        if (cand.empty()) {
            std::cerr << "Warning: No candidate satellites found for central index " << c << "\n";
            std::abort();
        }
        auto tr = transform(c, cand);

        int sat_count = 0;
        for (size_t k = 0; k < cand.size(); ++k) {
            int local_id = cand[k];
            if (group_label[local_id] != IDType(-1)) continue; // Ignore if already classified
            std::array<double,2> dens = density_contrast(c, tr.rel_dists[k], tr.rel_vels[k]);
            double P_M = dens[0];
            double B = dens[1]*scale;
            
            if (config.contrast) {
                if (P_M > B && masses_sorted[local_id] <= masses_sorted[c]) {
                    group_label[local_id] = (IDType)c;
                    if ((IDType)local_id != (IDType)c) { // Avoid classifying the central as its own satellite
                        classification[local_id] = 2; // 2 = satellite
                        sat_count++;
                    }
                }
            } else {
                if (config.vel_cut) {
                    if (tr.rel_dists[k] <= sel.R_h_group * halo_props[c].R_h &&
                        std::fabs(tr.rel_vels[k]) <= sel.V_vir_group * halo_props[c].V_vir / std::sqrt(2.0)){
                        group_label[local_id] = (IDType)c;
                        if ((IDType)local_id != (IDType)c) { // Avoid classifying the central as its own satellite
                            classification[local_id] = 2; // 2 = satellite
                            sat_count++;
                        }
                    }
                } else {
                    if (tr.rel_dists[k] <= sel.R_h_group * halo_props[c].R_h) {
                        group_label[local_id] = (IDType)c;
                        if ((IDType)local_id != (IDType)c) {
                            classification[local_id] = 2; // 2 = satellite
                            sat_count++;
                        }
                    }
                }
            }
        }
        group_label[c] = (IDType)c;
        if (sat_count == 0) classification[c] = 1; // 1 = isolated central
        else classification[c] = 0; // 0 = group central
        cand.clear();
    }
    
    summarize("After initial classification", group_label);
    // Apply satellite classification if requested
    if (config.sat_reclass) reassign_satellites(Rmax, periodic, scale);
    // Apply isolated central classification if requested
    if (config.iso_reclass) reassign_isolated(Rmax, periodic, scale);

    // Get the unique indices, which correspond to groups
    std::vector<IDType> labels = group_label;
    std::stable_sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    for (auto &id : labels) {
        if (id == IDType(-1)) {
            std::cerr << "Warning: Unassigned group label found in unique labels. This should not happen.\n";
            std::abort();
        }
    }
    
    // Map old (possibly now unused) labels
    std::unordered_map<IDType,IDType> old2new;
    old2new.reserve(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) old2new[labels[i]] = (IDType)i;

    // Temporary storage (may include groups that lost their central)
    std::vector<std::vector<IDType>> tmp_groups(labels.size());
    std::vector<IDType> tmp_centrals(labels.size(), IDType(-1));
    std::vector<double> tmp_halo_m(labels.size(), -1.0);

    // Fill memberships
    for (size_t loc = 0; loc < group_label.size(); ++loc) {
        IDType old_id = group_label[loc];
        if (old_id < 0) continue;
        size_t g = old2new[old_id];
        IDType global_id = groupcat_ids_sorted[loc];
        tmp_groups[g].push_back(global_id);
        if (classification[loc] == 0 || classification[loc] == 1) {
            tmp_centrals[g] = global_id;
            tmp_halo_m[g] = halo_props[loc].M_h;
        }
    }

    // Get rid of any groups lacking a central (can happen if an isolated central was reclassified)
    std::vector<std::vector<IDType>> group_indices;
    std::vector<IDType> central_global_indices;
    std::vector<double> halo_masses_final;

    group_indices.reserve(tmp_groups.size());
    central_global_indices.reserve(tmp_groups.size());
    halo_masses_final.reserve(tmp_groups.size());

    for (size_t g = 0; g < tmp_groups.size(); ++g) {
        if (tmp_groups[g].empty()) continue;
        if (tmp_centrals[g] == IDType(-1)) {
            // Skip orphan group (no surviving central -- reclassified as satellite)
            std::cerr << "Warning: Group " << g << " has no central. This should not happen.\n";
            std::abort();
        }
        // Ensure central first
        auto &members = tmp_groups[g];
        auto it = std::find(members.begin(), members.end(), tmp_centrals[g]);
        if (it != members.end() && it != members.begin()) {
            std::rotate(members.begin(), it, it + 1);
        }
        group_indices.push_back(std::move(members));
        central_global_indices.push_back(tmp_centrals[g]);
        halo_masses_final.push_back(tmp_halo_m[g]);
    }
    return std::make_tuple(group_indices, central_global_indices, halo_masses_final);
}

template<class D,class V>
std::tuple<std::vector<std::vector<IDType>>, std::vector<IDType>, std::vector<double>> 
GroupFinder<D,V>::run_once(
        const std::vector<double>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_box,
        const std::vector<Vec3>& velocities_pec,
        const Vec3& MW_pos_box, const Vec3& MW_vel_pec, 
        const double& Rmax, const double& scale, bool periodic) {
    initialize(masses_unsorted, groupcat_ids, positions_box, velocities_pec, MW_pos_box, MW_vel_pec, periodic);
    return classify(Rmax, scale, periodic);
}

template<class D,class V>
std::tuple<std::vector<std::vector<IDType>>, std::vector<IDType>, std::vector<double>> 
GroupFinder<D,V>::run_once_obs(
        const std::vector<double>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions,
        const std::vector<double>& velocities_los,
        const double& Rmax, const double& scale,
        bool periodic) {
    initialize_obs(masses_unsorted, groupcat_ids, positions, velocities_los);
    return classify(Rmax, scale, periodic);
}
};

// Explicit template instantiations

// 6D: 3D positions, 3D velocities
template class gf::GroupFinder<gf::Dist3D, gf::VelPeculiar3D>;
// 6D: 3D positions, line-of-sight peculiar velocity
template class gf::GroupFinder<gf::Dist3D, gf::VelPeculiar2D>;
 // 3D: 2D position (projected, not from Vincenty formula), line-of-sight velocity + Hubble flow
template class gf::GroupFinder<gf::DistPerp2D, gf::VelTotal>;
// 3D: 2D position from Vincenty formula, line-of-sight velocity + Hubble flow
// Direct analog to observation group finder
template class gf::GroupFinder<gf::DistProjectedRCTan, gf::VelTotal>;
// For classifying observational data
template class gf::GroupFinder<gf::DistObs, gf::VelObs>;