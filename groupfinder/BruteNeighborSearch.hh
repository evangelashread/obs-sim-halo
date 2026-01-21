#include "GroupFinderCore.hh"
#include <cmath>
#include <algorithm>

namespace gf {

inline std::vector<IDType> bruteforce_search(size_t central_id, const std::vector<Vec3>& all_positions,
                                         const std::vector<IDType>& cand_ids, double R_max, 
                                         double L, bool periodic = true, bool obs = false){
    /* Candidate IDs are the indices into all_positions to consider.
    For example, for satellite reclassification, we would use the local id of the satellite 
    as the central_id, and the local ids of the potential new central hosts for cand_ids.
    Then all_positions should be positions_sorted. */

    auto wrap = [L](double d){
        d -= L*std::round(d/L);
        return d;
    };
    std::vector<IDType> candidates;
    const auto& pos0 = all_positions[central_id];
    double R2 = R_max*R_max;
    for (size_t j : cand_ids) {
        if (obs) {
            // Spherical coordinates: d2 = |r-r'|^2 = (r^2 + r'^2 - 2 r r' (cos theta cos theta' + sin theta sin theta' cos(phi-phi')))
            double d2 = all_positions[j][0]*all_positions[j][0] + pos0[0]*pos0[0] 
            - 2.0*all_positions[j][0]*pos0[0]*(std::cos(all_positions[j][1])*std::cos(pos0[1]) 
            + std::sin(all_positions[j][1])*std::sin(pos0[1])*std::cos(all_positions[j][2]-pos0[2]));
            // If the distance is within the cutoff radius, add to neighbors
            if (d2 <= R2) candidates.push_back((IDType)j);
        } else { // Cartesian coordinates (simulation data)
            double dx = all_positions[j][0]-pos0[0];
            double dy = all_positions[j][1]-pos0[1];
            double dz = all_positions[j][2]-pos0[2];
            if (periodic) {
                dx = wrap(dx); dy = wrap(dy); dz = wrap(dz);
            }
            double d2 = dx*dx + dy*dy + dz*dz;
            // If the distance is within the cutoff radius, add to neighbors
            if (d2 <= R2) candidates.push_back((IDType)j);
        }
    }
    return candidates;
};
}