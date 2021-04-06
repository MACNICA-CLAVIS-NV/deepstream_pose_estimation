#include "pair_graph.hpp"
#include "cover_table.hpp"
//#include "munkres_algorithm.cpp"
#include "munkres_algorithm.hpp"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <stdio.h>
#include <vector>
#include <array>
#include <queue>
#include <cmath>

#define EPS 1e-6

template <class T>
using Vec1D = std::vector<T>;
template <class T>
using Vec2D = std::vector<Vec1D<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

void find_peaks(Vec1D<int> &counts_out, Vec3D<int> &peaks_out, void *cmap_data,
                NvDsInferDims &cmap_dims, float threshold, int window_size, int max_count);

Vec3D<float>
refine_peaks(Vec1D<int> &counts,
             Vec3D<int> &peaks, void *cmap_data, NvDsInferDims &cmap_dims,
             int window_size);

Vec3D<float>
paf_score_graph(void *paf_data, NvDsInferDims &paf_dims,
                Vec2D<int> &topology, Vec1D<int> &counts,
                Vec3D<float> &peaks, int num_integral_samples);

Vec3D<int>
assignment(Vec3D<float> &score_graph,
           Vec2D<int> &topology, Vec1D<int> &counts, float score_threshold, int max_count);

Vec2D<int>
connect_parts(
    Vec3D<int> &connections, Vec2D<int> &topology, Vec1D<int> &counts,
    int max_count);
