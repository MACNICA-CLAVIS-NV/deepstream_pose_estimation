template <class T>
using Vec1D = std::vector<T>;
template <class T>
using Vec2D = std::vector<Vec1D<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

// Helper method to subtract the minimum row from cost_graph
void subtract_minimum_row(Vec2D<float> &cost_graph, int nrows, int ncols);

// Helper method to subtract the minimum col from cost_graph
void subtract_minimum_column(Vec2D<float> &cost_graph, int nrows, int ncols);

void munkresStep1(Vec2D<float> &cost_graph, PairGraph &star_graph, int nrows, int ncols);

// Exits if '1' is returned
bool munkresStep2(const PairGraph &star_graph, CoverTable &cover_table);

bool munkresStep3(Vec2D<float> &cost_graph, const PairGraph &star_graph,
                  PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> &p,
                  int nrows, int ncols);

void munkresStep4(PairGraph &star_graph, PairGraph &prime_graph,
                  CoverTable &cover_table, std::pair<int, int> &p);

void munkresStep5(Vec2D<float> &cost_graph, const CoverTable &cover_table,
                  int nrows, int ncols);

void munkres_algorithm(Vec2D<float> &cost_graph, PairGraph &star_graph, int nrows,
              int ncols);
