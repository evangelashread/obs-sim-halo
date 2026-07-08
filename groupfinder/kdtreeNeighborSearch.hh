#pragma once
#include <vector>
#include <array>
#include <Aboria.h>
#include <algorithm>

namespace gf {

ABORIA_VARIABLE(id, IDType, "id")

using Vec3 = std::array<double, 3>;
using IDType = std::int64_t;

class AboriaNeighborBuilder {
    using Particles_t = Aboria::Particles<std::tuple<id>,3,std::vector,Aboria::Kdtree>;
    Particles_t particles;
    std::vector<IDType> indices; // Local indices corresponding to particles
    double min_L;
    double max_L;
    bool periodic_box;
public:
    AboriaNeighborBuilder(const std::vector<Vec3>& positions, const std::vector<IDType>& local_indices, double lower_bound, double box_size, bool periodic)
    : particles(positions.size()), indices(local_indices), min_L(lower_bound), max_L(box_size), periodic_box(periodic) {

        using position = Particles_t::position;
        // Fill particle positions (assumed to be cartesian)
        for (size_t i = 0; i < positions.size(); ++i) {
            Aboria::get<position>(particles[i]) = Aboria::vdouble3(positions[i][0], positions[i][1], positions[i][2]);
            Aboria::get<id>(particles[i]) = indices[i]; // Use local indices
        }
        // Initialize neighbor search
        Aboria::vdouble3 min = Aboria::vdouble3::Constant(min_L);
        Aboria::vdouble3 max = Aboria::vdouble3::Constant(max_L);
        Aboria::vbool3 pbc  = Aboria::vbool3::Constant(periodic_box);
        particles.init_neighbour_search(min,max,pbc);
    }

    std::vector<IDType> kdtree_search(size_t central_loc_id,
                                   const std::vector<Vec3>& positions,
                                   double search_radius) const {
        std::vector<IDType> candidates;
        if (central_loc_id < 0 || (size_t)central_loc_id >= positions.size()) return candidates;

        const auto& pos0 = positions[central_loc_id];
        candidates.reserve(32);

        for (auto it = Aboria::euclidean_search(particles.get_query(), Aboria::vdouble3(pos0[0],pos0[1],pos0[2]), search_radius); it != false; ++it) {
            IDType particle_id = Aboria::get<id>(*it);
            candidates.push_back(particle_id);
        }
        return candidates; // indices into positions
    }
};
}