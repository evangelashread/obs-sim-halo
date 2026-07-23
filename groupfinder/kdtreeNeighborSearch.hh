#pragma once
#include "Types.hh"
#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <Aboria.h>
#include <nanoflann/nanoflann.hpp>

namespace gf {

ABORIA_VARIABLE(id, IDType, "id")

class NeighborSearchBackend {
public:
    virtual ~NeighborSearchBackend() = default;
    virtual std::vector<IDType> kdtree_search(size_t central_loc_id,
                                               const std::vector<Vec3>& positions,
                                               double search_radius) const = 0;
};

// Use the Aboria kdtree (which does have periodic boundary support)
class AboriaBackend : public NeighborSearchBackend {
    using Particles_t = Aboria::Particles<std::tuple<id>, 3, std::vector, Aboria::Kdtree>;
    Particles_t particles;
public:
    AboriaBackend(const std::vector<Vec3>& positions,
                  const std::vector<IDType>& local_indices,
                  double lower_bound, double box_size, bool periodic, int leaf_size)
        : particles(positions.size())
    {
        using position = Particles_t::position;
        for (size_t i = 0; i < positions.size(); ++i) {
            Aboria::get<position>(particles[i]) = Aboria::vdouble3(positions[i][0], positions[i][1], positions[i][2]);
            Aboria::get<id>(particles[i]) = local_indices[i];
        }
        Aboria::vdouble3 min = Aboria::vdouble3::Constant(lower_bound);
        Aboria::vdouble3 max = Aboria::vdouble3::Constant(box_size);
        Aboria::vbool3 pbc = Aboria::vbool3::Constant(periodic);
        particles.init_neighbour_search(min, max, pbc, leaf_size);
    }

    std::vector<IDType> kdtree_search(size_t central_loc_id,
                                       const std::vector<Vec3>& positions,
                                       double search_radius) const override {
        std::vector<IDType> candidates;
        if (central_loc_id >= positions.size()) return candidates;
        const auto& pos0 = positions[central_loc_id];
        candidates.reserve(32);
        for (auto it = Aboria::euclidean_search(particles.get_query(), Aboria::vdouble3(pos0[0], pos0[1], pos0[2]), search_radius); it != false; ++it) {
            candidates.push_back(Aboria::get<id>(*it));
        }
        return candidates;
    }
};

// Use nanoflann, which is ideal for large data, but it has no periodic boundary support
struct PointCloudAdaptor {
    const std::vector<Vec3>& pts;
    explicit PointCloudAdaptor(const std::vector<Vec3>& pts_) : pts(pts_) {}
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx][dim]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
    PointCloudAdaptor, 3>;

class NanoflannBackend : public NeighborSearchBackend {
    PointCloudAdaptor cloud;
    KDTreeType index;
    std::vector<IDType> local_ids;
public:
    NanoflannBackend(const std::vector<Vec3>& positions,
                      const std::vector<IDType>& local_indices, 
                      int leaf_size, int n_threads)
        : cloud(positions),
          index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_threads)),
          local_ids(local_indices)
    {
        index.buildIndex();
    }

    std::vector<IDType> kdtree_search(size_t central_loc_id,
                                       const std::vector<Vec3>& positions,
                                       double search_radius) const override {
        std::vector<IDType> candidates;
        if (central_loc_id >= positions.size()) return candidates;
        const auto& pos0 = positions[central_loc_id];
        double query_pt[3] = {pos0[0], pos0[1], pos0[2]};

        std::vector<nanoflann::ResultItem<typename KDTreeType::IndexType, double>> matches;
        nanoflann::SearchParameters params;
        double radius_arg = search_radius * search_radius; // since we have configured with L2
        index.radiusSearch(query_pt, radius_arg, matches, params);
        candidates.reserve(matches.size());
        for (auto& m : matches) candidates.push_back(local_ids[m.first]);
        return candidates;
    }
};

// Default: use_nanoflann=false
class AboriaNeighborBuilder {
    std::unique_ptr<NeighborSearchBackend> backend;
public:
    AboriaNeighborBuilder(const std::vector<Vec3>& positions,
                           const std::vector<IDType>& local_indices,
                           double lower_bound, double box_size, bool periodic,
                           int leaf_size, int n_threads, bool use_nanoflann) {
        if (use_nanoflann && periodic) {
            throw std::runtime_error(
                "use_nanoflann=true is incompatible with periodic=true: nanoflann "
                "has no native periodic-boundary support. Set use_nanoflann=false "
                "for periodic runs, or handle periodicity beforehand with ghost particles.");
        }
        if (use_nanoflann) {
            backend = std::make_unique<NanoflannBackend>(positions, local_indices, leaf_size, n_threads);
        } else {
            backend = std::make_unique<AboriaBackend>(positions, local_indices, lower_bound, box_size, periodic, leaf_size);
        }
    }

    std::vector<IDType> kdtree_search(size_t central_loc_id,
                                       const std::vector<Vec3>& positions,
                                       double search_radius) const {
        return backend->kdtree_search(central_loc_id, positions, search_radius);
    }
};

} // namespace gf