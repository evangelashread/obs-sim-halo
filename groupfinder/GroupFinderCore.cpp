#include "GroupFinderCore.hh"
#include "kdtreeNeighborSearch.hh"
#include "BruteNeighborSearch.hh"
#include "MappedArray.hh"
#include "Types.hh"
#include <unistd.h>
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
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <string>
#include <omp.h>
#include <assert.h>

#ifdef USE_MALLOC
    #include <malloc.h>
#endif

namespace gf {

static inline double wrap(double x, double L) {
    /** 
     * @brief Wraps a value x to be within the bounds [-L/2, L/2],
     * where x may be an absolute or a relative distance
     */
    return x - L * std::round(x / L);
}

static inline Vec3 minimal(const Vec3& a, const Vec3& b, double L) {
    /** @brief Compute the minimal image distance vector between two points in a periodic box*/
    return {
        static_cast<FloatType>(wrap(a[0]-b[0],L)), static_cast<FloatType>(wrap(a[1]-b[1],L)), static_cast<FloatType>(wrap(a[2]-b[2],L))
    };
}

static inline double critical_density(double z, double p_crit_0, double omega_m) {
 return p_crit_0 * (omega_m * std::pow(1.0 + z, 3) + (1.0 - omega_m));
}

static inline double Hubble(double z, double H, double omega_m) {
   return H * std::sqrt(omega_m * std::pow(1.0 + z, 3) + (1.0 - omega_m));
}

static inline Vec3 celestial_to_cartesian(const Vec3& vec) {
    /** 
    @brief Convert celestial to cartesian coordinates. 
    Assumed that input vector is in the form (r, theta or dec [rad], phi or RA [rad]), where dec ranges 
    from [-pi/2, pi/2] and RA from [0, 2pi).
    */
    double x = vec[0] * std::cos(vec[1]) * std::cos(vec[2]);
    double y = vec[0] * std::cos(vec[1]) * std::sin(vec[2]);
    double z = vec[0] * std::sin(vec[1]);
    return {static_cast<FloatType>(x), static_cast<FloatType>(y), static_cast<FloatType>(z)};
}

static double behroozi_SMHM(double z, const double& x, double current_Mstar, const BehrooziParams& params) {
    /**
     * @brief Helper for the iterative solver find_Mh. Returns the difference between log Mstar calculated 
     * from the Behroozi et al. 2019 SMHM relation, and the current log Mstar for a given log Mpeak/M1 (x) 
     * and redshift (z). For full generality should make this generalizable to any SHMR...
     */
    
    double a = 1.0 / (1.0 + z);
    double a1 = a - 1.0;
    double lna = std::log(a);

    struct ZParams {
        double m_1, eps, alpha, beta, delta, gamma;
    };

    ZParams zp;

    zp.m_1 = params.M_0 + a1 * params.M_A - lna * params.M_LOGA + z * params.M_Z; // log10(M1/Msun)
    zp.eps = params.EPS_0 + a1 * params.EPS_A - lna * params.EPS_LOGA + z * params.EPS_Z;
    zp.alpha = params.ALPHA_0 + a1 * params.ALPHA_A - lna * params.ALPHA_LOGA + z * params.ALPHA_Z;
    zp.beta = params.BETA_0 + a1 * params.BETA_A + z * params.BETA_Z;
    zp.delta = params.DELTA_0;
    zp.gamma = std::pow(10, params.GAMMA_0 + a1 * params.GAMMA_A + z * params.GAMMA_Z);

    double x2 = x / zp.delta; // x = log10(Mpeak/M1)
    double logmstar_pred = zp.eps + zp.m_1
        - std::log10(std::pow(10, -zp.alpha * x) + std::pow(10, -zp.beta * x))
        + zp.gamma * std::exp(-0.5 * (x2 * x2)) ;
    
    return logmstar_pred - current_Mstar;
}

static inline double interp_lin(double x,
                                const std::vector<double>& x_arr,
                                const std::vector<double>& y_arr)
{
    // x_arr should be sorted in ascending order
    assert(x_arr.size() == y_arr.size() && x_arr.size() >= 2);
    // handle values outside bounds by clamping
    size_t i = 0;
    if (x <= x_arr.front()) { return y_arr.front(); } 
    else if (x >= x_arr.back()) { return y_arr.back(); } 
    else {
        auto it = std::upper_bound(x_arr.begin(), x_arr.end(), x); // first x_arr element > x
        i = size_t(it - x_arr.begin()) - 1; // index just below x
    }
    size_t j = i + 1;// index just above x
    // Linear interpolation
    if (x_arr[j] == x_arr[i]) { return y_arr[i]; }
    else {
        double t = (x - x_arr[i]) / (x_arr[j] - x_arr[i]);
        return (1.0 - t) * y_arr[i] + t * y_arr[j];
    }
}

static inline double bilinear_interp(double x, double y,
                                const std::vector<double>& x_arr,
                                const std::vector<double>& y_arr,
                                const std::vector<std::vector<double>>& z_arr)
{
    // x_arr and y_arr should be sorted in ascending order
    assert(x_arr.size() == z_arr.size() && y_arr.size() == z_arr[0].size());
    // Clip values outside the grid to bounds
    size_t i = 0;
    if (x <= x_arr.front()) { i = 0;} 
    else if (x >= x_arr.back()) { i = x_arr.size() - 2; } 
    else {
        auto it_x = std::upper_bound(x_arr.begin(), x_arr.end(), x);
        i = size_t(it_x - x_arr.begin()) - 1;
    }
    size_t k = 0;
    if (y <= y_arr.front()) { k = 0; } 
    else if (y >= y_arr.back()) { k = y_arr.size() - 2; } 
    else {
        auto it_y = std::upper_bound(y_arr.begin(), y_arr.end(), y);
        k = size_t(it_y - y_arr.begin()) - 1;
    }
    
    size_t j = i + 1;
    size_t l = k + 1;
    // don't want numeric slop if the interval's too small
    const double eps = 1e-15;
    double dx = x_arr[j] - x_arr[i];
    double dy = y_arr[l] - y_arr[k];
    double t = (std::fabs(dx) < eps) ? 0.0 : (x - x_arr[i]) / dx;
    double u = (std::fabs(dy) < eps) ? 0.0 : (y - y_arr[k]) / dy;
    
    return (1.0 - t) * (1.0 - u) * z_arr[i][k]
         + t * (1.0 - u) * z_arr[j][k]
         + (1.0 - t) * u* z_arr[i][l]
         + t * u * z_arr[j][l];
}

static double find_Mh(const double& z, double x0, double x1, double current_mass, 
                      const BehrooziParams& params, double tolerance = 1e-9, int max_iter = 100) {
    /**
     * @brief Secant root-finding method for inverting Behroozi SMHM relation
     * NOT CURRENTLY USED (ignore warning generation)
     */
    double f_x0 = behroozi_SMHM(z, x0, current_mass, params);
    double f_x1 = behroozi_SMHM(z, x1, current_mass, params);

    if (!std::isfinite(f_x0) || !std::isfinite(f_x1)) {
        return x1; // not finite before iteration even began
    }

    for (int i = 0; i < max_iter; ++i) {
        if (std::fabs(f_x1) < tolerance) {
            return x1;
        }

        if (std::fabs(x1 - x0) < tolerance) {
            std::cout << "x0 and x1 are too close together. Iteration stopped." << std::endl;
            return x1;
        }

        double denominator = (f_x1 - f_x0);

        if (std::fabs(denominator) < 1e-12) {
            std::cout << "Division by zero risk. Stopping iteration." << std::endl;
            return x1;
        }

        double x2 = x1 - f_x1 * (x1 - x0) / denominator;

        x0 = x1;
        f_x0 = f_x1;
        x1 = x2;
        f_x1 = behroozi_SMHM(z, x1, current_mass, params);

        if (!std::isfinite(x1) || !std::isfinite(f_x1)) {
            return x1; // became non finite in the middle of the iteration
        }
    }

    std::cout << "Max iterations reached without convergence. Last approximation: " << x1 << std::endl;
    return x1;
}

static inline double VelocityLOS(const Vec3& pos, const Vec3& vel) {
    /** @brief pos: in cartesian coordinates. vel: in x,y,z 3D vector form. */
    // Assume position and velocity is already defined relative to the observer
    double r2 = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2];
    if(r2==0.) return 0.0; // This case occurs if the galaxy is the MW
    double r = std::sqrt(r2);
    // Compute the normal vector for the line-of-sight distance
    Vec3 n = {static_cast<FloatType>(pos[0]/r), static_cast<FloatType>(pos[1]/r), static_cast<FloatType>(pos[2]/r)};
    // Compute the velocity of the galaxy projected along its line-of-sight normal vector
    double v_LOS = vel[0]*n[0] + vel[1]*n[1] + vel[2]*n[2];
    return v_LOS;
}

template<class D,class V>
HaloProps GroupFinder<D,V>::compute_halo_props(double z, double logMstar, double p_crit_0, const BehrooziParams& params, double omega_m) {
    /** @brief Precompute halo properties once per iteration of group_finding */
    //double a = 1.0 / (1.0 + z);
    //double a1 = a - 1.0;
    //double lna = std::log(a);
    //double sol = find_Mh(z, -5., 5., logMstar, params);
    //double M_1 = params.M_0 + a1 * params.M_A - lna * params.M_LOGA + z * params.M_Z; // log10(M1/Msun)
    //double M_h = std::pow(10., sol + M_1); // Msun
    double M_h = std::pow(10., Mh_from_Mstar(logMstar, z));
    double p_crit = critical_density(z, p_crit_0, omega_m);
    double R_h = std::cbrt((3.*M_h)/(4.*M_PI*200.*p_crit)); // Mpc
    double V_vir = std::sqrt(gf::GF_G * M_h / R_h); // km/s
    return {static_cast<FloatType>(M_h), static_cast<FloatType>(R_h), static_cast<FloatType>(V_vir)};
}

template<class D,class V>
void GroupFinder<D,V>::save_checkpoint(int phase, int in_progress_phase, size_t sub_progress) const {
    if (checkpoint_path.empty()) return;
    std::string tmp_path = checkpoint_path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out) {
            std::cerr << "Warning: could not open checkpoint file for writing: " << tmp_path << std::endl;
            return;
        }
        const uint32_t magic = 0x47464348; // "GFCH"
        const uint32_t version = 3;
        uint64_t n = group_label.size();
        int32_t phase32 = phase;
        int32_t in_progress32 = in_progress_phase;
        uint64_t sub_progress64 = static_cast<uint64_t>(sub_progress);
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&phase32), sizeof(phase32));
        out.write(reinterpret_cast<const char*>(&in_progress32), sizeof(in_progress32));
        out.write(reinterpret_cast<const char*>(&sub_progress64), sizeof(sub_progress64));
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(group_label.data()), n * sizeof(IDType));
        out.write(reinterpret_cast<const char*>(classification.data()), n * sizeof(classification[0]));
        if (!out) {
            std::cerr << "Warning: error writing checkpoint data -- previous checkpoint left untouched." << std::endl;
            return;
        }
    }
    // Rename only happens after the full write above succeeds, so if it crashes mid write we can fall back to the old checkpoint file
    if (std::rename(tmp_path.c_str(), checkpoint_path.c_str()) != 0) {
        std::cerr << "Warning: failed to finalize checkpoint (rename failed)." << std::endl;
    } else {
        std::cerr << "[CHECKPOINT] saved at phase=" << phase;
        if (in_progress_phase != 0) std::cerr << ", in_progress_phase=" << in_progress_phase << ", sub_progress=" << sub_progress;
        std::cerr << ", N=" << group_label.size() << std::endl;
    }
}

template<class D,class V>
bool GroupFinder<D,V>::load_checkpoint(int& phase_out, int& in_progress_phase_out, size_t& sub_progress_out) {
    phase_out = 0;
    in_progress_phase_out = 0;
    sub_progress_out = 0;
    if (checkpoint_path.empty()) return false;
    std::ifstream in(checkpoint_path, std::ios::binary);
    if (!in) return false; // no checkpoint yet

    uint32_t magic = 0, version = 0;
    int32_t phase32 = 0, in_progress32 = 0;
    uint64_t sub_progress64 = 0;
    uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || magic != 0x47464348) {
        std::cerr << "Warning: checkpoint file unreadable/corrupt; starting fresh." << std::endl;
        return false;
    }
    if (version != 3) { // just in case there's previoous versions of this floating around still
        std::cerr << "Warning: checkpoint file is an older/incompatible format (version " << version 
                << "); starting fresh. Delete " << checkpoint_path << " to silence this." << std::endl;
        return false;
    }
    in.read(reinterpret_cast<char*>(&phase32), sizeof(phase32));
    in.read(reinterpret_cast<char*>(&in_progress32), sizeof(in_progress32));
    in.read(reinterpret_cast<char*>(&sub_progress64), sizeof(sub_progress64));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in) {
        std::cerr << "Warning: checkpoint file unreadable/corrupt; starting fresh." << std::endl;
        return false;
    }
    if (n != group_label.size()) {
        std::cerr << "Warning: checkpoint galaxy count (" << n << ") != current dataset ("
                   << group_label.size() << "); ignoring checkpoint. Check the input file, maybe it changed." << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(group_label.data()), n * sizeof(IDType));
    in.read(reinterpret_cast<char*>(classification.data()), n * sizeof(classification[0]));
    if (!in) {
        std::cerr << "Warning: checkpoint data truncated; starting fresh." << std::endl;
        group_label.assign(n, IDType(-1));
        return false;
    }

    phase_out = phase32;
    in_progress_phase_out = in_progress32;
    sub_progress_out = static_cast<size_t>(sub_progress64);
    std::cerr << "[CHECKPOINT] resumed from phase=" << phase32;
    if (in_progress32 != 0) std::cerr << ", in_progress_phase=" << in_progress32 << ", sub_progress=" << sub_progress64;
    std::cerr << ", N=" << n << std::endl;
    return true;
}

void summarize(const std::string& label, const std::vector<IDType>& group_label) {
    std::unordered_map<IDType,IDType> group_sizes;
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        if (g == -1) {
            std::cerr << "Error: galaxy " << i << " is unassigned to any group." << std::endl;
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
              << " multi=" << (n_groups - n_single) << std::endl;
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

/* The below method is used if true 2D projected (NOT great-circle) distances are desired. */
double DistPerp2D::operator()(const Vec3& mw_c, const Vec3& mw_s, double L) const {

    // Compute the relative distance between the satellite and the central
    Vec3 d = minimal(mw_s, mw_c, L);
    
    double rc2 = mw_c[0]*mw_c[0]+mw_c[1]*mw_c[1]+mw_c[2]*mw_c[2];
    if(rc2==0.) return 0.0; // This case occurs if the central is the MW analogue

    // Compute the normal vector for the line-of-sight distance to the central
    double rc = std::sqrt(rc2);
    Vec3 n = {static_cast<FloatType>(mw_c[0]/rc), static_cast<FloatType>(mw_c[1]/rc), static_cast<FloatType>(mw_c[2]/rc)};

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

    if(rs2==0.) return 0.0; // unlikely to occur but avoids division by zero
    double rs_len = std::sqrt(rs2);
    double cos_theta = (r_cen_mw[0]*rs[0] + r_cen_mw[1]*rs[1] + r_cen_mw[2]*rs[2])/(rc_len*rs_len);

    // Clamp, only forward intersection allowed
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    if (cos_theta <= 0.0) return std::numeric_limits<double>::infinity();

    // haversine distance
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

    // haversine of theta
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
/* The below methods mimic velocity criteria implementation of an observational group finder,
and can be used in either simulation or observation data */
double VelTotal::operator()(const double& z_c, const double& z_s) const {          
    return gf::GF_C * (z_s - z_c) / (1.0 + z_c); // redshift space
}
 
double VelObs::operator()(const double& v_c, const double& v_s) const {
    return std::abs(v_s - v_c); // real space (peculiar LOS velocity)
}

/* The below methods are relevant for simulation based implementations selecting on peculiar velocity */
double VelPeculiar3D::operator()(const Vec3& v_c, const Vec3& v_s, double L) const {
    Vec3 rel_vel = {v_s[0]-v_c[0], v_s[1]-v_c[1], v_s[2]-v_c[2]};
    return std::sqrt(rel_vel[0]*rel_vel[0] + rel_vel[1]*rel_vel[1] + rel_vel[2]*rel_vel[2]);
}

double VelPeculiar2D::operator()(const Vec3& mw_c, const Vec3& mw_s,
                                const Vec3& v_c, const Vec3& v_s) const {
    return VelocityLOS(mw_s, v_s) - VelocityLOS(mw_c, v_c);
}

template<class D,class V>
double GroupFinder<D,V>::c_from_M(double logM, double z) const {
    if (tab_logM_conc_.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    if (tab_z_conc_.size() == 1) {
        double logc = interp_lin(logM, tab_logM_conc_, tab_logc_conc_[0]); // x in log10(M), y in log10(c)
        return std::pow(10.0, logc);
    } else {
        // bilinear interpolation in logM and z to get logc
        double logc = bilinear_interp(z, logM, tab_z_conc_, tab_logM_conc_, tab_logc_conc_);
        return std::pow(10.0, logc);
    }
}

template<class D,class V>
void GroupFinder<D,V>::set_conc_table(std::vector<double> logMstar_arr, std::vector<double> redshift,
                                      std::vector<std::vector<double>> logc_array) {
    const size_t N_M = logMstar_arr.size();
    const size_t N_z = redshift.size();

    if (logc_array.size() != N_z) {
        std::cerr << "Error: the number of vectors in the concentration table must equal the number of redshift bins." << std::endl;
        std::abort();
    }
    if (N_M < 2) {
        std::cerr << "Error: concentration table must have at least two entries" << std::endl;
        std::abort();
    }
    // Copy and build logs (assumes monotonic/unique already)
    tab_logM_conc_ = std::move(logMstar_arr);
    tab_z_conc_ = std::move(redshift);
    tab_logc_conc_ = std::move(logc_array);
}

template<class D,class V>
double GroupFinder<D,V>::Mh_from_Mstar(double logMstar, double z) const {
    if (tab_logMstar_smhm_.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    if (tab_z_smhm_.size() == 1) {
        double logMh = interp_lin(logMstar, tab_logMstar_smhm_, tab_logMh_smhm_[0]); // x in log10(M), y in log10(Mh)
        return logMh;
    } else {
        // bilinear interpolation in logM and z to get logc
        double logMh = bilinear_interp(z, logMstar, tab_z_smhm_, tab_logMstar_smhm_, tab_logMh_smhm_);
        return logMh;
    }
}

template<class D,class V>
void GroupFinder<D,V>::set_smhm_table(std::vector<double> logMstar_arr, std::vector<double> z_arr,
                                      std::vector<std::vector<double>> logMh_array) {
    const size_t N_M = logMstar_arr.size();
    const size_t N_z = z_arr.size();
    if (logMh_array.size() != N_z) {
        std::cerr << "Error: the number of vectors in the inverse SMHM table must equal the number of redshift bins." << std::endl;
        std::abort();
    }
    if (N_M < 2) {
        std::cerr << "Error: the inverse SMHM lookup table must have at least two entries" << std::endl;
        std::abort();
    }
    tab_z_smhm_ = std::move(z_arr);
    tab_logMstar_smhm_ = std::move(logMstar_arr);
    tab_logMh_smhm_ = std::move(logMh_array);
}

template<class D,class V>
double GroupFinder<D,V>::z_from_D(double d) const {
    if (tab_z_D_.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double z = interp_lin(d, tab_D_, tab_z_D_); // x in cMpc, y is z
    return z;
}

template<class D,class V>
double GroupFinder<D,V>::D_from_z(double z) const {
    if (tab_D_.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double d = interp_lin(z, tab_z_D_, tab_D_); // x is z, y in cMpc
    return d;
}

template<class D,class V>
void GroupFinder<D,V>::set_z_dist_table(std::vector<double> z_arr, std::vector<double> D_arr) {
    const size_t N_D = D_arr.size();
    const size_t N_z = z_arr.size();
    if (N_D < 2) {
        std::cerr << "Error: redshift-distance table must have at least two entries." << std::endl;
        std::abort();
    }
    // Copy and build logs (assumes monotonic/unique already)
    tab_z_D_.resize(N_z);
    tab_D_.resize(N_D);
    
    tab_z_D_ = z_arr;
    tab_D_ = D_arr;
}

// This will generally find the relative distance and velocity between two galaxies
template<class D,class V>
void GroupFinder<D,V>::transform(size_t central_local, size_t point_local, double& rel_dist, double& rel_vel, double& R_c_out) const {
   
    if (!config.obs) { // Check that this is simulation data
        const auto& mw_c = MWcoords[central_local];
        const auto& mw_s = MWcoords[point_local];
        const auto& v_c = velocities_sorted[central_local];
        const auto& v_s = velocities_sorted[point_local];
        const auto& z_c = total_redshifts[central_local];
        const auto& z_s = total_redshifts[point_local];
        if constexpr (std::is_same_v<D,DistObs>) {
            std::cerr << "Error: Distances must not be of type DistObs when using the 3D/2D distance methods." << std::endl;
            std::abort();
        } else {
            rel_dist = dist_method(mw_c,mw_s,L);
        }
        if constexpr (std::is_same_v<V,VelPeculiar3D>) {
            rel_vel = vel_method(v_c,v_s,L);
        } else if constexpr (std::is_same_v<V,VelPeculiar2D>) {
            rel_vel = vel_method(mw_c,mw_s,v_c,v_s);
        } else if constexpr (std::is_same_v<V,VelTotal>) {
            rel_vel = vel_method(z_c,z_s);
        }
        R_c_out = 0.0; // unused when not in obs mode
    } else { // Observational data
        double z_c = 0., v_c = 0.;
        double R_c = positions_sorted[central_local][0];
        if (config.use_distance) {
            v_c  = velocities_sorted_obs[central_local];
        } else {
            z_c = total_redshifts[central_local];
        }
        const auto& Dec_c = positions_sorted[central_local][1];
        const auto& RA_c = positions_sorted[central_local][2];
        
        const auto& Dec_s = positions_sorted[point_local][1];
        const auto& RA_s = positions_sorted[point_local][2];
        double z_s = 0., v_s = 0.;
        if (config.use_distance) {
            v_s = velocities_sorted_obs[point_local];
        } else {
            z_s = total_redshifts[point_local];
        }
        if constexpr (std::is_same_v<V,VelObs>) {
            rel_vel = vel_method(v_c, v_s);
        } else if constexpr (std::is_same_v<V,VelTotal>) {
            rel_vel = vel_method(z_c, z_s);
        } else {
            std::cerr << "Error: Velocities must be of type VelObs or VelTotal when using the observational distance method." << std::endl;
            std::abort();
        }
        if constexpr (std::is_same_v<D,DistObs>) {
            rel_dist = dist_method(RA_c, Dec_c, RA_s, Dec_s, R_c);
            R_c_out = R_c;
        } else {
            std::cerr << "Error: Distances must be of type DistObs when using the observational distance method." << std::endl;
            std::abort();
        }
    }
}

template<class D,class V>
TransformOutput GroupFinder<D,V>::transform_against_satellites(size_t central_local, const std::vector<IDType>& local_indices) const {
    TransformOutput tr;
    tr.rel_dists.resize(local_indices.size());
    tr.rel_vels.resize(local_indices.size());
    tr.R_cen.resize(local_indices.size());
    for (size_t i = 0; i < local_indices.size(); ++i) {
        transform(central_local, local_indices[i], tr.rel_dists[i], tr.rel_vels[i], tr.R_cen[i]);
    }
    return tr;
}

template<class D,class V>
TransformOutput GroupFinder<D,V>::transform_against_centrals(size_t point_local, const std::vector<IDType>& central_candidates) const {
    TransformOutput tr;
    tr.rel_dists.resize(central_candidates.size());
    tr.rel_vels.resize(central_candidates.size());
    tr.R_cen.resize(central_candidates.size());
    for (size_t i = 0; i < central_candidates.size(); ++i) {
        transform(central_candidates[i], point_local, tr.rel_dists[i], tr.rel_vels[i], tr.R_cen[i]);
    }
    return tr;
}

template<class D,class V>
std::array<double, 2> GroupFinder<D,V>::density_contrast(IDType local_c_id, double trans_dist, double rel_vel) {
    // Get V_vir, M200 and r200 from halo props
    double V_vir = halo_props[local_c_id].V_vir;
    double M_h = halo_props[local_c_id].M_h;
    double R_200 = halo_props[local_c_id].R_h;
    double z = total_redshifts[local_c_id];

    // Compute concentration from halo mass
    double c = c_from_M(std::log10(M_h), z);

    double r_s = R_200/c;
    double sigma_v = V_vir/std::sqrt(2.); // units of km/s
    double x = trans_dist / r_s;

    auto p = [&](double rel_vel) -> double {
        return (1./(std::sqrt(2.*M_PI)*sigma_v)) * std::exp(-(rel_vel*rel_vel) / (2.*sigma_v*sigma_v));
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
    double B = rho_200_rho_crit * (4.*R_200*Hubble(z, H, OMEGA_M))/(3.*sigma_v); // dimensionless
    double P_M = Hubble(z, H, OMEGA_M) * Sigma_R_rho_gal * p(rel_vel);
    return {P_M, B};
}

template<class D,class V>
void GroupFinder<D,V>::reassign_satellites(double search_radius, bool periodic, const double& scale, size_t resume_from) {
    // Implementation for reassigning satellites
    assert(classification.size() == positions_sorted.size());
    size_t n_central = 0, n_satellite = 0;
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0 || classification[i] == 1) ++n_central;
        else if (classification[i] == 2) ++n_satellite;
    }
    std::vector<IDType> central_indices; // indices into positions_sorted
    std::vector<Vec3> central_positions;
    std::vector<IDType> satellite_indices; // indices into positions_sorted
    central_indices.reserve(n_central);
    central_positions.reserve(n_central);
    satellite_indices.reserve(n_satellite);
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0 || classification[i] == 1) { // 0 = group central, 1 = isolated central
            central_indices.push_back((IDType)i);
            if (config.tree_search && config.obs) { 
                central_positions.push_back(cartesian_from_RA_Dec[i]); 
            } else {
                central_positions.push_back(positions_sorted[i]);
            }
        } else if (classification[i] == 2) {
            satellite_indices.push_back((IDType)i);
        } else {
            std::cerr << "Error: Galaxy without a classification found." << std::endl;
            std::abort();
        }
    }

    int reclassified = 0;
    size_t N = satellite_indices.size();

    if (resume_from > N) resume_from = 0; // stale or mismatched checkpoint -- don't skip past the end
    IDType classified = static_cast<IDType>(resume_from);
    const size_t chunk_size = (checkpoint_interval > 0) ? checkpoint_interval : N;

    omp_set_num_threads(config.n_threads);
    for (size_t chunk_start = resume_from; chunk_start < N; chunk_start += chunk_size) {
        size_t chunk_end = std::min(chunk_start + chunk_size, N);

        #pragma omp parallel
        {
        std::vector<IDType> cand;
        std::vector<double> ratios;
        std::vector<double> P_M_cands;
        std::vector<IDType> new_cands;
        cand.reserve(64);
        ratios.reserve(64);
        P_M_cands.reserve(64);
        new_cands.reserve(64);

        #pragma omp for schedule(dynamic, 64) reduction(+:reclassified)
        for (size_t s = chunk_start; s < chunk_end; ++s) {
            IDType local_s_id = satellite_indices[s]; // index into positions_sorted
            IDType local_c_id = group_label[(size_t)local_s_id];
            if (config.tree_search) { // Returns local indices
                double d_T = sel.R_h_group * halo_props[(size_t)local_c_id].R_h; // get the radius of the halo this satellite has been assigned to
                double search_radius_3d = 0.0;
                double los_margin = sel.V_vir_group * halo_props[(size_t)local_c_id].V_vir * (1.0 + total_redshifts[(size_t)local_s_id]) / (std::sqrt(2.0) * Hubble(total_redshifts[(size_t)local_s_id], H, OMEGA_M)); // max LOS offset the velocity cut allows 
                if (!config.obs) {
                    if (config.dim == 6) {
                        search_radius_3d = d_T; // spherical in 6D
                    } else { 
                        search_radius_3d = std::sqrt(los_margin*los_margin + d_T*d_T);
                    }
                    cand = tree->kdtree_search((size_t)local_s_id, positions_sorted, config.BUFFER * search_radius_3d); // returns value from central_indices
                } else {
                    cand = tree->kdtree_search((size_t)local_s_id, cartesian_from_RA_Dec, config.BUFFER * std::sqrt(los_margin*los_margin + d_T*d_T));
                }
            } else { // Returns local indices
                if (search_radius <= 0.0) {
                    cand = central_indices; // All centrals are candidates
                } else {
                    cand = bruteforce_search((size_t)local_s_id, positions_sorted, central_indices, search_radius, L, periodic, config.obs); // returns value from central_indices
                }
            }
            if (cand.empty()) {
                std::cerr << "Error: No candidate centrals found for satellite index " << local_s_id << std::endl;
                std::abort();
            }
            ratios.clear();
            P_M_cands.clear();
            new_cands.clear();

            auto tr = transform_against_centrals((size_t)local_s_id, cand);

            for (size_t k = 0; k < cand.size(); ++k) {
                size_t local_c_id = cand[k]; // index into positions_sorted
                // Using one tree for everything means our search returns ALL nearby galaxies, not just centrals 
                if (classification[local_c_id] != 0 && classification[local_c_id] != 1) continue;

                double rel_dist = tr.rel_dists[k];
                double rel_vel = tr.rel_vels[k];
                double R_h_cand = halo_props[local_c_id].R_h;
                double V_vir_cand = halo_props[local_c_id].V_vir;
                double ratio = (rel_dist / (sel.R_h_group * R_h_cand))*(rel_dist / (sel.R_h_group * R_h_cand)) 
                                + (rel_vel / (sel.V_vir_group * V_vir_cand / std::sqrt(2.0)))*(rel_vel / (sel.V_vir_group * V_vir_cand / std::sqrt(2.0)));
                                // ?? another typo in the appendix because R_h_group is 1 not 2 ??
                double mass_cand = masses_sorted[local_c_id];
                
                if (config.contrast) {
                    std::array<double,2> dens = density_contrast(local_c_id, rel_dist, rel_vel);
                    double P_M = dens[0];
                    double B = dens[1]*scale; // same typo as before ?? 
                    if (P_M >= B && masses_sorted[(size_t)local_s_id] < mass_cand) {
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
                    std::cerr << "Error: No candidate centrals passed criteria for satellite index " << local_s_id << std::endl;
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
                if (ratios.empty()) {
                    // The satellite remains in its current group
                    // Unlikely to occur; a satellite usually meets all criteria above to have been assigned to a group in part 1
                    std::cerr << "Error: No candidate centrals passed criteria for satellite index " << local_s_id << std::endl;
                    continue;
                } else if (ratios.size() == 1) {
                    IDType local_c_id = new_cands[0];
                    // If the assigned central is different from the one assigned in part 1, count it as reclassified
                    if (group_label[(size_t)local_s_id] != local_c_id) {
                        reclassified += 1;
                        group_label[(size_t)local_s_id] = local_c_id;
                    }
                } else {
                    // More than one candidate remains, find the one with the smallest ratio
                    auto min_it = std::min_element(ratios.begin(), ratios.end());
                    int min_index = std::distance(ratios.begin(), min_it);
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
            IDType local_count;
            #pragma omp atomic capture
            local_count = ++classified;
            if (local_count % config.chunk_readout == IDType(0)) {
                #pragma omp critical(sat_progress_print)
                {
                    double percent_classified = 100.0 * (static_cast<double>(local_count) / static_cast<double>(N));
                    std::cout << "\r" << percent_classified << " percent of satellites have been reviewed for reclassification. " << std::flush;
                }
            }
        } // end of for loop through this chunk's satellites
        } // end of parallel region; safe to checkpoint

        // phase=1 (initial classification is the last FULLY completed phase boundary); in_progress_phase=2, sub_progress=chunk_end
        save_checkpoint(1, 2, chunk_end);
    }
    std::cout << " " << std::endl;

    std::unordered_map<IDType,int> group_sizes;
    group_sizes.reserve(central_indices.size());
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        if (g == IDType(-1)) {
            std::cerr << "Error: galaxy " << i << " is unassigned to any group." << std::endl;
            std::abort();
        } else ++group_sizes[g];
    }
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        int sz = group_sizes[g];
        if (sz == 1) {
            if (classification[i] == 2) {
                std::cerr << "Error: Satellite assigned to a group of size 1 found. This should not happen." << std::endl;
                std::abort();
            } else {
                classification[i] = 1; // isolated central
                group_label[i] = (IDType)i; // ensure self-label
            }
        } else if (sz > 1) {
            if (classification[i] == 1) classification[i] = 0; // reclassify isolated central to group central
        } 
    }
    std::cout << reclassified << " satellites reclassified to different groups" << std::endl;
    summarize("After satellite reclassification", group_label);
}

template<class D,class V>
void GroupFinder<D,V>::reassign_isolated_phase_a(double search_radius, bool periodic, const double& scale, size_t resume_from) {
    /** @brief Isolated-central reclassification, PHASE A.
        Group centrals (already sorted in descending mass order by construction of initialize/initialize_obs)
        can take nearby isolated centrals directly as satellites ("class 3")
    */
    std::vector<IDType> group_central_indices;
    std::vector<Vec3> group_central_positions;
    std::vector<IDType> isolated_central_indices;
    assert(classification.size() == positions_sorted.size());

    size_t n_group_central = 0, n_isolated = 0;
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0) ++n_group_central;
        else if (classification[i] == 1) ++n_isolated;
    }
    group_central_indices.reserve(n_group_central);
    group_central_positions.reserve(n_group_central);
    isolated_central_indices.reserve(n_isolated);

    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0) { // 0 = group central
            group_central_indices.push_back(i);
            if (config.tree_search && config.obs) {
                group_central_positions.push_back(cartesian_from_RA_Dec[i]);
            } else {
                group_central_positions.push_back(positions_sorted[i]);
            }
        } else if (classification[i] == 1) { // 1 = isolated central
            isolated_central_indices.push_back(i);
        }
    }

    size_t N = group_central_indices.size();
    int reclassified = 0;

    if (resume_from > N) resume_from = 0; // stale/mismatched checkpoint
    IDType classified = static_cast<IDType>(resume_from);
    const size_t chunk_size = (checkpoint_interval > 0) ? checkpoint_interval : N;

    for (size_t chunk_start = resume_from; chunk_start < N; chunk_start += chunk_size) {
        size_t chunk_end = std::min(chunk_start + chunk_size, N);

        std::vector<IDType> cand;
        std::vector<IDType> filtered;
        cand.reserve(64);
        filtered.reserve(64);

        for (size_t c = chunk_start; c < chunk_end; ++c) {
            IDType local_c_id = group_central_indices[c];
            if (config.tree_search) {
                double d_T = sel.R_h_iso * halo_props[(size_t)local_c_id].R_h;
                double search_radius_3d = 0.0;
                double los_margin = sel.V_vir_iso * halo_props[(size_t)local_c_id].V_vir * (1.0 + total_redshifts[(size_t)local_c_id]) / (std::sqrt(2.0) * Hubble(total_redshifts[(size_t)local_c_id], H, OMEGA_M)); // max LOS offset the velocity cut allows, 
                if (!config.obs) {
                    if (config.dim == 6) {
                        search_radius_3d = d_T; // spherical in 6D
                    } else { 
                        search_radius_3d = std::sqrt(los_margin*los_margin + d_T*d_T);
                    }
                    cand = tree->kdtree_search((size_t)local_c_id, positions_sorted, config.BUFFER * search_radius_3d); // returns value from central_indices
                } else {
                    cand = tree->kdtree_search((size_t)local_c_id, cartesian_from_RA_Dec, config.BUFFER * std::sqrt(los_margin*los_margin + d_T*d_T));
                }
            } else { // tree_search == false
                if (search_radius <= 0.0) {
                    cand = isolated_central_indices;
                } else {
                    cand = bruteforce_search((size_t)local_c_id, positions_sorted, isolated_central_indices, search_radius, L, periodic, config.obs);
                }
            }
            if (cand.empty()) {
                continue;
            }
            filtered.clear();
            for (IDType id : cand) {
                if (classification[(size_t)id] == 1) filtered.push_back(id);
            }
            std::swap(cand, filtered);

            auto tr = transform_against_satellites((size_t)local_c_id, cand);
            for (size_t k = 0; k < cand.size(); ++k) {
                size_t local_i_id = cand[k]; // index into positions_sorted
                // Flag every isolated central within R_h_iso * R_h and V_vir_iso * V_vir as a potential satellite
                if (tr.rel_dists[k] <= sel.R_h_iso * halo_props[(size_t)local_c_id].R_h &&
                    std::fabs(tr.rel_vels[k]) <= sel.V_vir_iso * halo_props[(size_t)local_c_id].V_vir / std::sqrt(2.0)){
                    group_label[local_i_id] = local_c_id;
                    classification[local_i_id] = 3; // 3 = isolated central that was reassigned as a satellite
                    reclassified++;
                }
            }
            cand.clear();

            IDType local_count = ++classified;
            if (classified % config.chunk_readout == IDType(0)) {
                double percent_classified = 100.0 * (static_cast<double>(local_count) / static_cast<double>(n_isolated));
                std::cout << "\r" << percent_classified << " percent done reviewing isolated centrals for reclassification. " << std::flush;
            }
        } // end of for loop through this chunk's isolated centrals

        // phase=2 (satellite reclassification is the last FULLY completed phase boundary); in_progress_phase=3, sub_progress=chunk_end
        save_checkpoint(2, 3, chunk_end);
    }
    std::cout << " " << std::endl;
    
    std::cout << reclassified << " isolated centrals flagged for potential reclassification as satellites" << std::endl;
}

template<class D,class V>
void GroupFinder<D,V>::reassign_isolated_phase_b(double search_radius, bool periodic, const double& scale, size_t resume_from) {
    /** @brief Isolated-central reclassification, PHASE B.
        Galaxies filtered in phase A ("class 3") are re-evaluated against the full group central population.
        Same parallel structure as reassign_satellites.
     */
    assert(classification.size() == positions_sorted.size());
    size_t n_central = 0, n_class3 = 0;
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0 || classification[i] == 1) ++n_central;
        else if (classification[i] == 3) ++n_class3;
    }
    std::vector<IDType> central_indices; // indices into positions_sorted
    std::vector<Vec3> central_positions;
    std::vector<IDType> class3_indices; // indices into positions_sorted
    central_indices.reserve(n_central);
    central_positions.reserve(n_central);
    class3_indices.reserve(n_class3);
    for (size_t i = 0; i < classification.size(); ++i) {
        if (classification[i] == 0) { // 0 = group central
            central_indices.push_back((IDType)i);
            if (config.tree_search && config.obs) { 
                central_positions.push_back(cartesian_from_RA_Dec[i]); 
            } else {
                central_positions.push_back(positions_sorted[i]);
            }
        } else if (classification[i] == 3) {
            class3_indices.push_back((IDType)i);
        }
    }

    int reclassified = 0;
    size_t N = class3_indices.size();

    if (resume_from > N) resume_from = 0; // stale or mismatched checkpoint
    IDType classified = static_cast<IDType>(resume_from);
    const size_t chunk_size = (checkpoint_interval > 0) ? checkpoint_interval : N;

    int sat_count = 0;

    omp_set_num_threads(config.n_threads);
    for (size_t chunk_start = resume_from; chunk_start < N; chunk_start += chunk_size) {
        size_t chunk_end = std::min(chunk_start + chunk_size, N);

        #pragma omp parallel
        {
        std::vector<IDType> cand;
        std::vector<double> ratios;
        std::vector<double> P_M_cands;
        std::vector<IDType> new_cands;
        cand.reserve(64);
        ratios.reserve(64);
        P_M_cands.reserve(64);
        new_cands.reserve(64);

        #pragma omp for schedule(dynamic, 64) reduction(+:reclassified)
        for (size_t s = chunk_start; s < chunk_end; ++s) {
            IDType local_s_id = class3_indices[s]; // index into positions_sorted
            IDType local_c_id = group_label[(size_t)local_s_id];
            if (config.tree_search) { // Returns local indices
                double d_T = sel.R_h_iso * halo_props[(size_t)local_c_id].R_h; // get the radius of the halo this satellite has been assigned to
                double search_radius_3d = 0.0;
                double los_margin = sel.V_vir_iso * halo_props[(size_t)local_c_id].V_vir * (1.0 + total_redshifts[(size_t)local_s_id]) / (std::sqrt(2.0) * Hubble(total_redshifts[(size_t)local_s_id], H, OMEGA_M)); // max LOS offset the velocity cut allows 
                if (!config.obs) {
                    if (config.dim == 6) {
                        search_radius_3d = d_T; // spherical in 6D
                    } else { 
                        search_radius_3d = std::sqrt(los_margin*los_margin + d_T*d_T);
                    }
                    cand = tree->kdtree_search((size_t)local_s_id, positions_sorted, config.BUFFER * search_radius_3d); // returns value from central_indices
                } else {
                    cand = tree->kdtree_search((size_t)local_s_id, cartesian_from_RA_Dec, config.BUFFER * std::sqrt(los_margin*los_margin + d_T*d_T));
                }
            } else { // Returns local indices
                if (search_radius <= 0.0) {
                    cand = central_indices; // All centrals are candidates
                } else {
                    cand = bruteforce_search((size_t)local_s_id, positions_sorted, central_indices, search_radius, L, periodic, config.obs); // returns value from central_indices
                }
            }
            if (cand.empty()) {
                std::cerr << "Error: No candidate centrals found for distant satellite index " << local_s_id << std::endl;
                std::abort();
            }
            ratios.clear();
            P_M_cands.clear();
            new_cands.clear();

            auto tr = transform_against_centrals((size_t)local_s_id, cand);

            for (size_t k = 0; k < cand.size(); ++k) {
                size_t local_c_id = cand[k]; // index into positions_sorted
                // Using one tree for everything means our search returns ALL nearby galaxies, not just centrals 
                if (classification[local_c_id] != 0) continue;

                double rel_dist = tr.rel_dists[k];
                double rel_vel = tr.rel_vels[k];
                double R_h_cand = halo_props[local_c_id].R_h;
                double V_vir_cand = halo_props[local_c_id].V_vir;
                double ratio = (rel_dist / (sel.R_h_iso * R_h_cand))*(rel_dist / (sel.R_h_iso * R_h_cand)) 
                                + (rel_vel / (sel.V_vir_iso * V_vir_cand / std::sqrt(2.0)))*(rel_vel / (sel.V_vir_iso * V_vir_cand / std::sqrt(2.0)));
                double mass_cand = masses_sorted[local_c_id];
                
                if (config.contrast) {
                    std::array<double,2> dens = density_contrast(local_c_id, rel_dist, rel_vel);
                    double P_M = dens[0];
                    double B = dens[1]*scale * (sel.R_h_iso / sel.V_vir_iso);
                    if (P_M >= B && masses_sorted[(size_t)local_s_id] < mass_cand) {
                        new_cands.push_back(cand[k]);
                        P_M_cands.push_back(P_M);
                    }
                } else {
                    if (config.vel_cut) {
                        if (rel_dist <= sel.R_h_iso * R_h_cand && masses_sorted[(size_t)local_s_id] < mass_cand
                            && std::fabs(rel_vel) <= sel.V_vir_iso * V_vir_cand / std::sqrt(2.0)) {
                            ratios.push_back(ratio);
                            new_cands.push_back(cand[k]);
                        }
                    } else {
                        if (rel_dist <= sel.R_h_iso * R_h_cand && masses_sorted[(size_t)local_s_id] < mass_cand) {
                            ratios.push_back(ratio);
                            new_cands.push_back(cand[k]);
                        }
                    }
                }
            }

            if (config.contrast) {
                int max_index = -1;
                if (P_M_cands.empty()) {
                    // No candidate centrals passed criteria for this distant satellite galaxy, which would be due to the mass criteria
                    // keep as an isolated central
                    classification[(size_t)local_s_id] = 1;
                    group_label[(size_t)local_s_id] = local_s_id;
                    continue;
                } else if (P_M_cands.size() == 1) {
                    max_index = 0;
                    IDType local_c_id = new_cands[max_index];
                    if (group_label[(size_t)local_s_id] != local_c_id) {
                        reclassified += 1;
                        group_label[(size_t)local_s_id] = local_c_id;
                    }
                    classification[(size_t)local_s_id] = 2; // safe to reassign this class 3 back to a regular old satellite
                    sat_count += 1;
                } else {
                    auto max_it = std::max_element(P_M_cands.begin(), P_M_cands.end());
                    max_index = std::distance(P_M_cands.begin(), max_it);
                    IDType local_c_id = new_cands[max_index];
                    if (group_label[(size_t)local_s_id] != local_c_id) {
                        reclassified += 1;
                        group_label[(size_t)local_s_id] = local_c_id;
                    }
                    classification[(size_t)local_s_id] = 2;
                    sat_count += 1;
                }
            } else {
                if (ratios.empty()) {
                    // No candidate centrals passed criteria for this distant satellite index; keep isolated
                    classification[(size_t)local_s_id] = 1;
                    group_label[(size_t)local_s_id] = local_s_id;
                    continue;
                } else if (ratios.size() == 1) {
                    IDType local_c_id = new_cands[0];
                    if (group_label[(size_t)local_s_id] != local_c_id) {
                        reclassified += 1;
                        group_label[(size_t)local_s_id] = local_c_id;
                    }
                    classification[(size_t)local_s_id] = 2;
                    sat_count += 1;
                } else {
                    auto min_it = std::min_element(ratios.begin(), ratios.end());
                    int min_index = std::distance(ratios.begin(), min_it);
                    IDType local_c_id = new_cands[min_index];
                    if (group_label[(size_t)local_s_id] != local_c_id) {
                        reclassified += 1;
                        group_label[(size_t)local_s_id] = local_c_id;
                    }
                    classification[(size_t)local_s_id] = 2;
                    sat_count += 1;
                }
            }
            cand.clear();
            IDType local_count;
            #pragma omp atomic capture
            local_count = ++classified;
            if (local_count % config.chunk_readout == IDType(0)) {
                #pragma omp critical(sat_progress_print)
                {
                    double percent_classified = 100.0 * (static_cast<double>(local_count) / static_cast<double>(N));
                    std::cout << "\r" << percent_classified << " percent of distance satellites have been reviewed for reclassification. " << std::flush;
                }
            }
        } // end of for loop through this chunk's satellites
        } // end of parallel region; safe to checkpoint

        // phase=3 (isolated-central phase A is the last FULLY completed phase boundary); in_progress_phase=4, sub_progress=chunk_end.
        save_checkpoint(3, 4, chunk_end);
    }
    std::cout << " " << std::endl;

    std::unordered_map<IDType,int> group_sizes;
    group_sizes.reserve(central_indices.size());
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        if (g == IDType(-1)) {
            std::cerr << "Error: galaxy " << i << " is unassigned to any group." << std::endl;
            std::abort();
        } else ++group_sizes[g];
    }
    for (size_t i = 0; i < group_label.size(); ++i) {
        IDType g = group_label[i];
        int sz = group_sizes[g];
        if (sz == 1) {
            if (classification[i] == 2) {
                std::cerr << "Error: Satellite assigned to a group of size 1 found. This should not happen." << std::endl;
                std::abort();
            } else {
                classification[i] = 1; // isolated central
                group_label[i] = (IDType)i; // ensure self-label
            }
        }
    }
    std::cout << sat_count << " isolated centrals reclassified as satellites" << std::endl;
    std::cout << reclassified << " of these were classified to a different group than in phase A" << std::endl;
    summarize("After isolated central reclassification", group_label);
}

template<class D,class V>
void GroupFinder<D,V>::initialize(const std::vector<FloatType>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_box,
        const std::vector<Vec3>& velocities_pec,
        const Vec3& MW_pos_box,
        const Vec3& MW_vel_pec,
        bool periodic) {

    // Sort the galaxies by descending mass
    const double P_CRIT_0 = (3. * H*H) / (8. * M_PI * gf::GF_G); // M_sun/Mpc^3
    size_t N = masses_unsorted.size();
    std::vector<IDType> mass_order(N);
    std::iota(mass_order.begin(), mass_order.end(), IDType(0));
    std::stable_sort(mass_order.begin(), mass_order.end(),
              [&](IDType a,IDType b){return masses_unsorted[a] > masses_unsorted[b];});

    std::string base = sorted_cache_prefix.empty()
        ? ("/tmp/gf_scratch_" + std::to_string(::getpid()))
        : sorted_cache_prefix;
    auto path = [&](const char* suffix) { return base + suffix; };

    masses_sorted = MappedArray<FloatType>::create(path("_masses.bin"), N);
    total_redshifts = MappedArray<FloatType>::create(path("_redshifts.bin"), N);
    groupcat_ids_sorted = MappedArray<IDType>::create(path("_ids.bin"), N);
    positions_sorted = MappedArray<Vec3>::create(path("_positions.bin"), N);
    velocities_sorted = MappedArray<Vec3>::create(path("_velocities.bin"), N);
    MWcoords = MappedArray<Vec3>::create(path("_mwcoords.bin"), N);
    halo_props = MappedArray<HaloProps>::create(path("_haloprops.bin"), N);
    classification.resize(N);

    IDType initialized = IDType(0);
    for(size_t i = 0; i < N; ++i) {
        initialized += 1;
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
        if (periodic) {
            d[0] = wrap(d[0],L); d[1] = wrap(d[1],L); d[2] = wrap(d[2],L);
        }
        double d2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        double dist = 0., cosmo_redshift = 0.;
        if (d2 > 0.) { 
            dist = std::sqrt(d2); 
            cosmo_redshift = z_from_D(dist);
        }
        if (config.dim == 3) { // we are working in redshift space and need contribution from peculiar velocities
            // Get v along the line of sight
            double v_los = VelocityLOS(d, v);
            double z_pec = v_los / gf::GF_C; // peculiar velocities are assumed non-relativistic
            total_redshifts[i] = (1 + cosmo_redshift)*(1 + z_pec) - 1.;
        } else { // work in real space
            total_redshifts[i] = cosmo_redshift;
        }

        MWcoords[i] = d;
        velocities_sorted[i] = v;
        halo_props[i] = compute_halo_props(total_redshifts[i],
                                            masses_unsorted[mass_order[i]], 
                                            P_CRIT_0,
                                            gf::BehrooziParams(),
                                            OMEGA_M);
        groupcat_ids_sorted[i] = groupcat_ids[mass_order[i]];
        if (initialized % config.chunk_readout == IDType(0)) { 
            double percent_initialized = 100.0 * (static_cast<double>(initialized) / static_cast<double>(N)); 
            std::cout << "\r" << percent_initialized << " percent of galaxies have been prepared for group classification." << std::flush; 
        }
    }
    std::cout << "Done calculating halo properties and initializing sorted arrays." << std::endl;
    
    mass_order.clear();

    // now these are read only from file, which is good if swap memory is a problem
    masses_sorted.finalize_read_only();
    total_redshifts.finalize_read_only();
    groupcat_ids_sorted.finalize_read_only();
    positions_sorted.finalize_read_only();
    velocities_sorted.finalize_read_only();
    MWcoords.finalize_read_only();
    halo_props.finalize_read_only();
}

template<class D,class V>
void GroupFinder<D,V>::initialize_obs(const std::vector<FloatType>& masses_unsorted,
        const std::vector<IDType>& groupcat_ids,
        const std::vector<Vec3>& positions_unsorted,
        const std::vector<FloatType>& velocities_los) {

    const double P_CRIT_0 = (3. * H*H) / (8. * M_PI * gf::GF_G); // M_sun/Mpc^3
    // Sort the galaxies by descending mass
    size_t N = masses_unsorted.size();
    std::vector<IDType> mass_order(N);
    std::iota(mass_order.begin(), mass_order.end(), IDType(0));
    std::stable_sort(mass_order.begin(), mass_order.end(),
              [&](IDType a,IDType b){return masses_unsorted[a] > masses_unsorted[b];});

    std::string base = sorted_cache_prefix.empty()
        ? ("/tmp/gf_scratch_" + std::to_string(::getpid()))
        : sorted_cache_prefix;
    auto path = [&](const char* suffix) { return base + suffix; };

    masses_sorted = MappedArray<FloatType>::create(path("_masses.bin"), N);
    groupcat_ids_sorted = MappedArray<IDType>::create(path("_ids.bin"), N);
    positions_sorted = MappedArray<Vec3>::create(path("_positions.bin"), N);
    total_redshifts = MappedArray<FloatType>::create(path("_redshifts.bin"), N);
    halo_props = MappedArray<HaloProps>::create(path("_haloprops.bin"), N);
    classification.resize(N);

    if (config.use_distance) {
        velocities_sorted_obs = MappedArray<FloatType>::create(path("_velocities_obs.bin"), N);
    }

    IDType initialized = IDType(0);
    for(size_t i = 0; i < N; ++i) {
        initialized += IDType(1);
        masses_sorted[i] = masses_unsorted[mass_order[i]];
        if (config.use_distance) { // need to calculate cosmological redshifts from comoving distance (peculiar velocity is separated)
            positions_sorted[i]  = positions_unsorted[mass_order[i]]; // (r, dec, RA)
            velocities_sorted_obs[i] = velocities_los[mass_order[i]];
            total_redshifts[i] = z_from_D(positions_unsorted[mass_order[i]][0]);
        } else { // compute comoving distance from redshift (includes contribution from peculiar velocity; fully in redshift space)
            total_redshifts[i] = positions_unsorted[mass_order[i]][0]; // z
            positions_sorted[i][0] = D_from_z(positions_unsorted[mass_order[i]][0]);
            positions_sorted[i][1] = positions_unsorted[mass_order[i]][1]; // copy dec
            positions_sorted[i][2] = positions_unsorted[mass_order[i]][2]; // copy RA
        }
        halo_props[i] = compute_halo_props(total_redshifts[i],
                                            masses_unsorted[mass_order[i]],
                                            P_CRIT_0,
                                            gf::BehrooziParams(),
                                            OMEGA_M); // means that halo props are sorted in descending mass order by construction
        groupcat_ids_sorted[i] = groupcat_ids[mass_order[i]];
        if (initialized % config.chunk_readout == IDType(0)) { 
            double percent_initialized = 100.0 * (static_cast<double>(initialized) / static_cast<double>(N)); 
            std::cout << "\r" << percent_initialized << " percent of galaxies have been prepared for group classification." << std::flush; 
        }
    }
    std::cout << "Done calculating halo properties and initializing sorted arrays." << std::endl;

    if (config.tree_search) {
        std::string radec_path = sorted_cache_prefix.empty()
            ? ("/tmp/gf_scratch_" + std::to_string(::getpid()) + "_radec.bin")
            : (sorted_cache_prefix + "_radec.bin");
        cartesian_from_RA_Dec = MappedArray<Vec3>::create(radec_path, N);
    }
    mass_order.clear();

    // make these read only to mitigate swap memory issues since they don't get modified again
    masses_sorted.finalize_read_only();
    groupcat_ids_sorted.finalize_read_only();
    positions_sorted.finalize_read_only();
    total_redshifts.finalize_read_only();
    halo_props.finalize_read_only();
    if (config.use_distance) velocities_sorted_obs.finalize_read_only();
}

template<class D,class V>
std::tuple<GroupsResult, std::vector<IDType>, std::vector<FloatType>>
GroupFinder<D,V>::classify(const double& search_radius, const double& scale, const bool& periodic) {
    /** @brief The work horse of the entire group finder */

    size_t N = masses_sorted.size();

    // Only needed as a candidate list for the non-tree-search path 
    std::vector<IDType> all_indices_tmp;
    if (!config.tree_search) {
        all_indices_tmp.resize(N);
        std::iota(all_indices_tmp.begin(), all_indices_tmp.end(), IDType(0));
    }

    // If kdtree, initialize the NearestNeighborBuilder class
    // Rebuild the tree upon every call to the class (fine for now because this is <10 min build even at billion scale...but should prob cache these at some point)
    if (config.tree_search) {
        if (!config.obs) {
            tree = std::make_unique<NearestNeighborBuilder>(positions_sorted, 0.0, L, periodic, config.leaf_size, config.n_threads, config.use_nanoflann);
        } else {
            // Convert dec (1st index) and RA (2nd index) to cartesian coords and build tree from it
            for (size_t v = 0; v < N; ++v) {
                Vec3 vec = {positions_sorted[v][0], positions_sorted[v][1], positions_sorted[v][2]};
                cartesian_from_RA_Dec[v] = celestial_to_cartesian(vec);
            }
            tree = std::make_unique<NearestNeighborBuilder>(cartesian_from_RA_Dec, -L, L, periodic, config.leaf_size, config.n_threads, config.use_nanoflann);
        }
        std::cout << "Done building kd tree for initial classification." << std::endl;
    }

    group_label.assign(N, IDType(-1)); // -1 means unassigned
    IDType classified = IDType(0);

    int resume_phase = 0;
    int in_progress_phase = 0;
    size_t sub_progress = 0;
    load_checkpoint(resume_phase, in_progress_phase, sub_progress);

    if (resume_phase < 1) {
        for(size_t c = 0; c < N; ++c) {
            classified += IDType(1);
            if (group_label[c] != IDType(-1)) continue; // The central has already been assigned a group

            // Find the candidate satellites
            std::vector<IDType> cand;
            if (config.tree_search) { // Returns local indices
                double d_T = sel.R_h_group * halo_props[c].R_h; // max transverse distance the criterion allows
                double search_radius_3d = 0.0;
                double los_margin = sel.V_vir_group * halo_props[c].V_vir * (1.0 + total_redshifts[c]) / (std::sqrt(2.0) * Hubble(total_redshifts[c], H, OMEGA_M)); // max LOS offset the velocity cut allows, 
                if (!config.obs) {
                    if (config.dim == 6) {
                        search_radius_3d = d_T; // spherical in 6D
                    } else { 
                        search_radius_3d = std::sqrt(los_margin*los_margin + d_T*d_T);
                    }
                    cand = tree->kdtree_search(c, positions_sorted, config.BUFFER * search_radius_3d); 
                } else {
                    cand = tree->kdtree_search(c, cartesian_from_RA_Dec, config.BUFFER * std::sqrt(los_margin*los_margin + d_T*d_T));
                }
            } else { // Returns local indices
                if (search_radius <= 0.0) {
                    cand = all_indices_tmp; // All galaxies are candidates
                } else {
                    cand = bruteforce_search(c, positions_sorted, all_indices_tmp, search_radius, L, periodic, config.obs);
                }
            }
            if (cand.empty()) {
                std::cerr << "Error: No candidate satellites found for central index " << c << std::endl;
                std::abort();
            }
            auto tr = transform_against_satellites(c, cand);

            int sat_count = 0;
            for (size_t k = 0; k < cand.size(); ++k) {
                IDType local_id = cand[k];
                if (group_label[local_id] != IDType(-1)) continue; // Ignore if already classified
                
                if (config.contrast) {
                    std::array<double,2> dens = density_contrast(c, tr.rel_dists[k], tr.rel_vels[k]);
                    double P_M = dens[0];
                    double B = dens[1]*scale; // ?? typo in A24 in Shread+26 ?? it should be 1 not 1/3
                    if (P_M >= B && masses_sorted[local_id] < masses_sorted[c]) {
                        group_label[local_id] = (IDType)c;
                        if ((IDType)local_id != (IDType)c) { // Avoid classifying the central as its own satellite
                            classification[local_id] = 2; // 2 = satellite
                            sat_count++;
                        }
                    }
                } else {
                    if (config.vel_cut) {
                        if (tr.rel_dists[k] <= sel.R_h_group * halo_props[c].R_h && masses_sorted[local_id] < masses_sorted[c]
                            && std::fabs(tr.rel_vels[k]) <= sel.V_vir_group * halo_props[c].V_vir / std::sqrt(2.0)){
                            group_label[local_id] = (IDType)c;
                            if ((IDType)local_id != (IDType)c) { // Avoid classifying the central as its own satellite
                                classification[local_id] = 2; // 2 = satellite
                                sat_count++;
                            }
                        }
                    } else {
                        if (tr.rel_dists[k] <= sel.R_h_group * halo_props[c].R_h && masses_sorted[local_id] < masses_sorted[c]) {
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
            if (classified % config.chunk_readout == IDType(0)) { 
                double percent_classified = 100.0 * (static_cast<double>(classified) / static_cast<double>(N)); 
                std::cout << "\r" << percent_classified << " percent of galaxies have been initially classified." << std::flush; 
            }
            if (classified % checkpoint_interval == 0) {
                save_checkpoint(0);
            }
            cand.clear();
        }
        std::cout << " " << std::endl;
        
        summarize("After initial classification", group_label);
        save_checkpoint(1);
    } else {
        std::cerr << "[CHECKPOINT] main classification loop already complete, skipping." << std::endl;
    }
    #ifdef USE_MALLOC
        malloc_trim(0);
    #endif
    // Apply satellite classification if requested
    if (config.sat_reclass && resume_phase < 2) {
        size_t sat_resume_from = (in_progress_phase == 2) ? sub_progress : 0;
        reassign_satellites(search_radius, periodic, scale, sat_resume_from);
        save_checkpoint(2);
        #ifdef USE_MALLOC
            malloc_trim(0);
        #endif
    } else if (config.sat_reclass) {
        std::cerr << "[CHECKPOINT] satellite reclassification already complete, skipping." << std::endl;
    }
    /* Apply isolated central classification if requested.
        resume_phase values: 
            <2 satellite reclass not done (handled above)
            2 = satellite reclass done, isolated phase A not yet complete
            3 = isolated phase A complete, phase B not done yet
            4 = isolated reclassification (both phases) fully complete
    */
    if (config.iso_reclass && resume_phase < 4) {
        if (resume_phase < 3) {
            size_t phaseA_resume_from = (in_progress_phase == 3) ? sub_progress : 0;
            reassign_isolated_phase_a(search_radius, periodic, scale, phaseA_resume_from);
            save_checkpoint(3); // phase A now fully complete, phase B not yet started
            #ifdef USE_MALLOC
                malloc_trim(0);
            #endif
        } else {
            std::cerr << "[CHECKPOINT] isolated-central phase A already complete, skipping." << std::endl;
        }
        size_t phaseB_resume_from = (in_progress_phase == 4) ? sub_progress : 0;
        reassign_isolated_phase_b(search_radius, periodic, scale, phaseB_resume_from);
        save_checkpoint(4); // isolated reclassification (both phases) fully complete
        #ifdef USE_MALLOC
            malloc_trim(0);
        #endif
    } else if (config.iso_reclass) {
        std::cerr << "[CHECKPOINT] isolated reclassification already complete, skipping." << std::endl;
    }

    // Get the unique indices, which correspond to groups
    std::vector<IDType> labels = group_label;
    std::stable_sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    for (auto &id : labels) {
        if (id == IDType(-1)) {
            std::cerr << "Error: Unassigned group label found in unique labels. This should not happen." << std::endl;
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
    std::vector<FloatType> tmp_halo_m(labels.size(), -1.0);

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
    std::vector<FloatType> halo_masses_final;

    group_indices.reserve(tmp_groups.size());
    central_global_indices.reserve(tmp_groups.size());
    halo_masses_final.reserve(tmp_groups.size());

    for (size_t g = 0; g < tmp_groups.size(); ++g) {
        if (tmp_groups[g].empty()) continue;
        if (tmp_centrals[g] == IDType(-1)) {
            // Abort if we have a group with no surviving central
            std::cerr << "Error: Group " << g << " has no central. This should not happen." << std::endl;
            std::abort();
        }
        // Ensure central is first
        auto &members = tmp_groups[g];
        auto it = std::find(members.begin(), members.end(), tmp_centrals[g]);
        if (it != members.end() && it != members.begin()) {
            std::rotate(members.begin(), it, it + 1);
        }
        group_indices.push_back(std::move(members));
        central_global_indices.push_back(tmp_centrals[g]);
        halo_masses_final.push_back(tmp_halo_m[g]);
    }
    // For faster I/O, flatten the ragged group_indices array
    GroupsResult res;
    res.offsets.resize(group_indices.size() + 1, 0);
    for (size_t g = 0; g < group_indices.size(); ++g) {
        res.offsets[g+1] = res.offsets[g] + static_cast<IDType>(group_indices[g].size());
    }
    res.member_ids.reserve(static_cast<size_t>(res.offsets.back()));
    for (auto& members : group_indices) {
        res.member_ids.insert(res.member_ids.end(), members.begin(), members.end());
    }
    std::cout << "Done preparing final group indices, central indices, and halo mass vectors." << std::endl;
    if (!checkpoint_path.empty()) {
        std::remove(checkpoint_path.c_str()); // run finished successfully. delete so a future unrelated run never resumes a stale state
    }
    return std::make_tuple(std::move(res), std::move(central_global_indices), std::move(halo_masses_final));
}

template<class D,class V>
std::tuple<GroupsResult, std::vector<IDType>, std::vector<FloatType>> 
GroupFinder<D,V>::run_once(
        std::vector<FloatType>& masses_unsorted,
        std::vector<IDType>& groupcat_ids,
        std::vector<Vec3>& positions_box,
        std::vector<Vec3>& velocities_pec,
        const Vec3& MW_pos_box, const Vec3& MW_vel_pec, 
        const double& search_radius, const double& scale, bool periodic) {
    /**
     * @brief High-level function for running the sim group finder.
     * Gets called directly by external code.
     */
    initialize(masses_unsorted, groupcat_ids, positions_box, velocities_pec, MW_pos_box, MW_vel_pec, periodic);
    masses_unsorted.clear();
    groupcat_ids.clear();
    positions_box.clear();
    velocities_pec.clear();
    masses_unsorted.shrink_to_fit();
    groupcat_ids.shrink_to_fit();
    positions_box.shrink_to_fit();
    velocities_pec.shrink_to_fit();
    #ifdef USE_MALLOC
        malloc_trim(0);
    #endif
    return classify(search_radius, scale, periodic);
}

template<class D,class V>
std::tuple<GroupsResult, std::vector<IDType>, std::vector<FloatType>> 
GroupFinder<D,V>::run_once_obs(
        std::vector<FloatType>& masses_unsorted,
        std::vector<IDType>& groupcat_ids,
        std::vector<Vec3>& positions,
        std::vector<FloatType>& velocities_los,
        const double& search_radius, const double& scale,
        bool periodic) {
    /**
     * @brief High-level function for running the observational group finder.
     */
    initialize_obs(masses_unsorted, groupcat_ids, positions, velocities_los);
    masses_unsorted.clear();
    groupcat_ids.clear();
    positions.clear();
    velocities_los.clear();
    masses_unsorted.shrink_to_fit();
    groupcat_ids.shrink_to_fit();
    positions.shrink_to_fit();
    velocities_los.shrink_to_fit();
    #ifdef USE_MALLOC
        malloc_trim(0);
    #endif
    return classify(search_radius, scale, periodic);
}
};

// 6D: 3D positions, 3D velocities
template class gf::GroupFinder<gf::Dist3D, gf::VelPeculiar3D>;
// 6D: 3D positions, line-of-sight peculiar velocity
template class gf::GroupFinder<gf::Dist3D, gf::VelPeculiar2D>;
 // 3D: 2D position (projected, not from Vincenty formula), line-of-sight peculiar velocity + Hubble flow
template class gf::GroupFinder<gf::DistPerp2D, gf::VelTotal>;
// 3D: 2D position from Vincenty formula, line-of-sight peculiar velocity + Hubble flow
// Direct analog to observation group finder
template class gf::GroupFinder<gf::DistProjectedRCTan, gf::VelTotal>;
// For classifying observational data with comoving distance and peculiar velocity
template class gf::GroupFinder<gf::DistObs, gf::VelObs>;
// For classifying observational data with redshift, RA and Dec only
template class gf::GroupFinder<gf::DistObs, gf::VelTotal>;