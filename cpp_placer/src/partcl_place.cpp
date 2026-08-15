// partcl_place.cpp - C++20 mixed-size placer for the par.tcl intern challenge.
//
// WHAT THE SCORER ACTUALLY MEASURES
// ---------------------------------
// `wirelength_attraction_loss()` costs each edge as
//
//     alpha * log( exp(|dx|/alpha) + exp(|dy|/alpha) ),   alpha = 0.1
//
// The docstring calls this a "smooth approximation of Manhattan distance", but
// alpha*logsumexp(dx/alpha, dy/alpha) is a smooth approximation of *max*, not of
// the sum: it sits within alpha*ln2 = 0.069 of max(|dx|,|dy|) everywhere. The
// objective is therefore Chebyshev (L-inf) wirelength. This solver optimises
// that function exactly rather than optimising Manhattan and hoping.
//
// Pins are also placed at `cell_pos + pin_offset` with the offset drawn from
// [0, w] x [0, h], while the overlap check treats `cell_pos` as the cell centre.
// So a cell's pin cloud is its body translated by (+w/2, +h/2). The solver never
// assumes pins are centred; it optimises the true pin positions, which lets big
// and small cells interleave their pin clouds.
//
// PIPELINE
// --------
//   1. GlobalPlace   Adam on smoothed-L-inf WL + pairwise overlap-area penalty,
//                    auto-scaled penalty multiplier, decaying step, uniform-grid
//                    neighbour search, positions projected into the die.
//   2. Macros        Either legalised from the global placement, or shelf-packed
//                    (first-fit decreasing height) into one edge band of the die.
//   3. RowLegalize   Every standard cell has height exactly 1.0, so they are
//                    packed into a unit-pitch row grid bounded by the die, with
//                    macros inserted as blocked intervals. Legality is
//                    structural: rows cannot overlap in y, and intervals inside
//                    a row are kept disjoint by construction.
//   4. Detailed      Single-cell relocation to the exact 1-D optimum of the real
//                    objective, snapped to the nearest free slot.
//   5. Multi-start   Sweep die aspect ratio x macro arrangement until the time
//                    budget is spent; keep the best legal result.
//
// Positions are finally rounded to float32 (the dtype the harness stores them
// in) and checked with a sweep-line implementation of the scorer's own overlap
// predicate, so "zero overlap" is verified, not assumed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

constexpr double kAlpha = 0.1;  // must match wirelength_attraction_loss()
// Legality margin. float32 resolves ~6e-5 at |coord| = 1000, so 5e-3 leaves
// ~80x headroom after the harness stores positions as float32.
constexpr double kGap = 5e-3;
constexpr double kInf = 1e300;
constexpr double kMacroH = 1.5;  // anything taller than a standard cell row

inline double EdgeCost(double dx, double dy) {
  dx = std::fabs(dx);
  dy = std::fabs(dy);
  const double hi = dx > dy ? dx : dy;
  const double lo = dx > dy ? dy : dx;
  return hi + kAlpha * std::log1p(std::exp(-(hi - lo) / kAlpha));
}

// Cost of one edge as seen from cell i: EdgeCost(x_i - x_j + ddx, y_i - y_j + ddy)
struct Adj {
  int j;
  double ddx, ddy;
};

struct Problem {
  int n = 0;
  std::vector<double> w, h, x, y;
  std::vector<std::vector<Adj>> adj;
  std::vector<int> ea, eb;       // inter-cell edges, by cell index
  std::vector<double> edx, edy;  // pin-offset deltas per edge
  double const_cost = 0.0;       // intra-cell edges: fixed, but scored
  int total_edges = 0;
  std::vector<int> macros, stds;
  double area = 0.0;
  double side = 1.0;  // sqrt(total cell area): the scorer's normaliser
};

struct Die {
  double W = 1, H = 1;
};

// ---------------------------------------------------------------------------
// objective
// ---------------------------------------------------------------------------

double TotalWL(const Problem& P) {
  double s = P.const_cost;
  for (size_t e = 0; e < P.ea.size(); ++e) {
    const int i = P.ea[e], j = P.eb[e];
    s += EdgeCost(P.x[i] - P.x[j] + P.edx[e], P.y[i] - P.y[j] + P.edy[e]);
  }
  return s;
}

double NormalizedWL(const Problem& P) {
  if (P.total_edges == 0) return 0.0;
  return (TotalWL(P) / P.total_edges) / P.side;
}

inline double CellCost(const Problem& P, int i, double px, double py) {
  double s = 0.0;
  for (const Adj& a : P.adj[i]) {
    s += EdgeCost(px - P.x[a.j] + a.ddx, py - P.y[a.j] + a.ddy);
  }
  return s;
}

// ---------------------------------------------------------------------------
// uniform grid for overlap neighbour search
// ---------------------------------------------------------------------------

struct Grid {
  double cw = 1, ch = 1, x0 = 0, y0 = 0;
  int nx = 1, ny = 1;
  std::vector<int> head, next;

  void Build(const Problem& P, const std::vector<int>& ids, double cw_, double ch_) {
    cw = cw_;
    ch = ch_;
    double lo_x = kInf, hi_x = -kInf, lo_y = kInf, hi_y = -kInf;
    for (int i : ids) {
      lo_x = std::min(lo_x, P.x[i]);
      hi_x = std::max(hi_x, P.x[i]);
      lo_y = std::min(lo_y, P.y[i]);
      hi_y = std::max(hi_y, P.y[i]);
    }
    if (ids.empty()) lo_x = hi_x = lo_y = hi_y = 0;
    x0 = lo_x - cw;
    y0 = lo_y - ch;
    nx = std::clamp(static_cast<int>((hi_x - lo_x) / cw) + 3, 1, 3000);
    ny = std::clamp(static_cast<int>((hi_y - lo_y) / ch) + 3, 1, 3000);
    head.assign(static_cast<size_t>(nx) * ny, -1);
    next.assign(P.n, -1);
    for (int i : ids) {
      const int gx = std::clamp(static_cast<int>((P.x[i] - x0) / cw), 0, nx - 1);
      const int gy = std::clamp(static_cast<int>((P.y[i] - y0) / ch), 0, ny - 1);
      const size_t c = static_cast<size_t>(gy) * nx + gx;
      next[i] = head[c];
      head[c] = i;
    }
  }
};

// ---------------------------------------------------------------------------
// stage 1: analytic global placement
// ---------------------------------------------------------------------------

void GlobalPlace(Problem& P, const std::vector<char>& movable, int iters, const Die* die) {
  if (iters <= 0) return;
  const int n = P.n;
  std::vector<double> gx(n), gy(n), owx(n), owy(n);
  std::vector<double> mx(n, 0.0), my(n, 0.0), vx(n, 0.0), vy(n, 0.0);

  double max_std_w = 1.0;
  for (int i : P.stds) max_std_w = std::max(max_std_w, P.w[i]);
  const double gcw = 2.0 * max_std_w + 0.2;
  const double gch = 2.2;

  const double lr0 = 0.05 * P.side;
  const double lr1 = 0.02;  // ~2% of a standard-cell row at the end
  const double b1 = 0.9, b2 = 0.999, eps = 1e-9;
  Grid grid;

  for (int it = 0; it < iters; ++it) {
    std::fill(gx.begin(), gx.end(), 0.0);
    std::fill(gy.begin(), gy.end(), 0.0);
    std::fill(owx.begin(), owx.end(), 0.0);
    std::fill(owy.begin(), owy.end(), 0.0);

    for (size_t e = 0; e < P.ea.size(); ++e) {
      const int i = P.ea[e], j = P.eb[e];
      const double ux = P.x[i] - P.x[j] + P.edx[e];
      const double uy = P.y[i] - P.y[j] + P.edy[e];
      const double dx = std::fabs(ux), dy = std::fabs(uy);
      const double sigma = 1.0 / (1.0 + std::exp(-(dx - dy) / kAlpha));
      const double sx = (ux > 0 ? 1.0 : (ux < 0 ? -1.0 : 0.0)) * sigma;
      const double sy = (uy > 0 ? 1.0 : (uy < 0 ? -1.0 : 0.0)) * (1.0 - sigma);
      gx[i] += sx;
      gx[j] -= sx;
      gy[i] += sy;
      gy[j] -= sy;
    }

    auto pair_grad = [&](int i, int j) {
      const double ox = 0.5 * (P.w[i] + P.w[j]) - std::fabs(P.x[i] - P.x[j]);
      if (ox <= 0.0) return;
      const double oy = 0.5 * (P.h[i] + P.h[j]) - std::fabs(P.y[i] - P.y[j]);
      if (oy <= 0.0) return;
      const double fx = (P.x[i] >= P.x[j] ? 1.0 : -1.0) * oy;
      const double fy = (P.y[i] >= P.y[j] ? 1.0 : -1.0) * ox;
      owx[i] += fx;
      owx[j] -= fx;
      owy[i] += fy;
      owy[j] -= fy;
    };

    // macros are few: check them against everything
    for (int i : P.macros) {
      for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        if (P.h[j] > kMacroH && j < i) continue;  // macro-macro pair once only
        pair_grad(i, j);
      }
    }
    // standard cells against each other via the grid
    grid.Build(P, P.stds, gcw, gch);
    for (int i : P.stds) {
      const int gxi = std::clamp(static_cast<int>((P.x[i] - grid.x0) / grid.cw), 0, grid.nx - 1);
      const int gyi = std::clamp(static_cast<int>((P.y[i] - grid.y0) / grid.ch), 0, grid.ny - 1);
      for (int dy = -1; dy <= 1; ++dy) {
        const int cy = gyi + dy;
        if (cy < 0 || cy >= grid.ny) continue;
        for (int dx = -1; dx <= 1; ++dx) {
          const int cx = gxi + dx;
          if (cx < 0 || cx >= grid.nx) continue;
          for (int j = grid.head[static_cast<size_t>(cy) * grid.nx + cx]; j >= 0; j = grid.next[j]) {
            if (j <= i) continue;
            pair_grad(i, j);
          }
        }
      }
    }

    // Auto-scale the penalty against the wirelength gradient so one schedule
    // works across design sizes: mu ramps the penalty from advisory to dominant.
    double nw = 0.0, no = 0.0;
    for (int i = 0; i < n; ++i) {
      nw += gx[i] * gx[i] + gy[i] * gy[i];
      no += owx[i] * owx[i] + owy[i] * owy[i];
    }
    nw = std::sqrt(nw);
    no = std::sqrt(no);
    const double t = static_cast<double>(it) / std::max(1, iters - 1);
    const double mu = 0.3 * std::pow(200.0, t);
    const double lambda = (no > 1e-12) ? mu * nw / no : 0.0;
    const double lr = lr0 * std::pow(lr1 / lr0, t);

    const double bc1 = 1.0 - std::pow(b1, it + 1);
    const double bc2 = 1.0 - std::pow(b2, it + 1);
    for (int i = 0; i < n; ++i) {
      if (!movable[i]) continue;
      const double ggx = gx[i] + lambda * owx[i];
      const double ggy = gy[i] + lambda * owy[i];
      mx[i] = b1 * mx[i] + (1 - b1) * ggx;
      my[i] = b1 * my[i] + (1 - b1) * ggy;
      vx[i] = b2 * vx[i] + (1 - b2) * ggx * ggx;
      vy[i] = b2 * vy[i] + (1 - b2) * ggy * ggy;
      P.x[i] -= lr * (mx[i] / bc1) / (std::sqrt(vx[i] / bc2) + eps);
      P.y[i] -= lr * (my[i] / bc1) / (std::sqrt(vy[i] / bc2) + eps);
      if (die) {
        P.x[i] = std::clamp(P.x[i], -0.5 * die->W + 0.5 * P.w[i], 0.5 * die->W - 0.5 * P.w[i]);
        P.y[i] = std::clamp(P.y[i], -0.5 * die->H + 0.5 * P.h[i], 0.5 * die->H - 0.5 * P.h[i]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// stage 2: macro placement
// ---------------------------------------------------------------------------

bool MacroFits(const Problem& P, int i, double px, double py, const std::vector<int>& placed,
               const Die& die) {
  if (px - 0.5 * P.w[i] < -0.5 * die.W - 1e-9) return false;
  if (px + 0.5 * P.w[i] > 0.5 * die.W + 1e-9) return false;
  if (py - 0.5 * P.h[i] < -0.5 * die.H - 1e-9) return false;
  if (py + 0.5 * P.h[i] > 0.5 * die.H + 1e-9) return false;
  for (int j : placed) {
    if (std::fabs(px - P.x[j]) < 0.5 * (P.w[i] + P.w[j]) + kGap &&
        std::fabs(py - P.y[j]) < 0.5 * (P.h[i] + P.h[j]) + kGap) {
      return false;
    }
  }
  return true;
}

// Legalise macros as close as possible to the positions the global placer chose.
bool LegalizeMacrosGreedy(Problem& P, const std::vector<double>& tx, const std::vector<double>& ty,
                          const Die& die) {
  std::vector<int> order = P.macros;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return P.w[a] * P.h[a] > P.w[b] * P.h[b]; });
  std::vector<int> placed;
  for (int i : order) {
    const double cx = std::clamp(tx[i], -0.5 * die.W + 0.5 * P.w[i], 0.5 * die.W - 0.5 * P.w[i]);
    const double cy = std::clamp(ty[i], -0.5 * die.H + 0.5 * P.h[i], 0.5 * die.H - 0.5 * P.h[i]);
    double best = kInf, bx = 0, by = 0;
    auto consider = [&](double px, double py) {
      const double d = std::fabs(px - cx) + std::fabs(py - cy);
      if (d < best && MacroFits(P, i, px, py, placed, die)) {
        best = d;
        bx = px;
        by = py;
      }
    };
    consider(cx, cy);
    for (int j : placed) {
      const double sx = 0.5 * (P.w[i] + P.w[j]) + kGap;
      const double sy = 0.5 * (P.h[i] + P.h[j]) + kGap;
      for (int s = -1; s <= 1; s += 2) {
        consider(P.x[j] + s * sx, cy);
        consider(P.x[j] + s * sx, P.y[j]);
        consider(cx, P.y[j] + s * sy);
        consider(P.x[j], P.y[j] + s * sy);
        for (int s2 = -1; s2 <= 1; s2 += 2) consider(P.x[j] + s * sx, P.y[j] + s2 * sy);
      }
      // slide along the die edges too
      for (int s = -1; s <= 1; s += 2) {
        consider(P.x[j] + s * sx, -0.5 * die.H + 0.5 * P.h[i]);
        consider(P.x[j] + s * sx, 0.5 * die.H - 0.5 * P.h[i]);
        consider(-0.5 * die.W + 0.5 * P.w[i], P.y[j] + s * sy);
        consider(0.5 * die.W - 0.5 * P.w[i], P.y[j] + s * sy);
      }
    }
    if (best >= kInf) return false;
    P.x[i] = bx;
    P.y[i] = by;
    placed.push_back(i);
  }
  return true;
}

// Pack macros into one edge band of the die, first-fit decreasing height.
// Keeps the rest of the die as one contiguous region for the standard cells,
// which is what the (low pin density, high area) macros should give up.
bool ShelfPackMacros(Problem& P, const Die& die, const std::vector<int>& order, int corner,
                     bool vertical) {
  const double W = vertical ? die.H : die.W;
  const double H = vertical ? die.W : die.H;
  double cx = 0.0, cy = 0.0, shelf_h = 0.0;
  std::vector<double> lx(P.n), ly(P.n);
  for (int m : order) {
    const double w = vertical ? P.h[m] : P.w[m];
    const double h = vertical ? P.w[m] : P.h[m];
    if (w > W || h > H) return false;
    if (cx + w > W + 1e-9) {
      cx = 0.0;
      cy += shelf_h + kGap;
      shelf_h = 0.0;
    }
    if (cy + h > H + 1e-9) return false;
    lx[m] = cx + 0.5 * w;
    ly[m] = cy + 0.5 * h;
    cx += w + kGap;
    shelf_h = std::max(shelf_h, h);
  }
  for (int m : order) {
    double ux = lx[m], uy = ly[m];
    if (corner & 1) ux = W - ux;
    if (corner & 2) uy = H - uy;
    ux -= 0.5 * W;
    uy -= 0.5 * H;
    P.x[m] = vertical ? uy : ux;
    P.y[m] = vertical ? ux : uy;
  }
  return true;
}

// ---------------------------------------------------------------------------
// stage 3: row legalization for standard cells
// ---------------------------------------------------------------------------

struct Iv {
  double lo, hi;
  int cell;
};

struct Rows {
  double pitch = 1.0;
  double xlo = 0, xhi = 0;
  int rlo = 0, rhi = 0;
  std::unordered_map<int, std::vector<Iv>> occ;

  void Reset(const Die& die) {
    occ.clear();
    pitch = 1.0 + kGap;
    xlo = -0.5 * die.W;
    xhi = 0.5 * die.W;
    // A row of height 1 centred at r*pitch must fit inside the die.
    rlo = static_cast<int>(std::ceil((-0.5 * die.H + 0.5) / pitch - 1e-9));
    rhi = static_cast<int>(std::floor((0.5 * die.H - 0.5) / pitch + 1e-9));
  }

  double RowY(int r) const { return r * pitch; }
  int NearestRow(double yv) const {
    return std::clamp(static_cast<int>(std::lround(yv / pitch)), rlo, rhi);
  }

  void Insert(int r, double lo, double hi, int cell) {
    auto& v = occ[r];
    auto it = std::lower_bound(v.begin(), v.end(), lo,
                               [](const Iv& a, double b) { return a.lo < b; });
    v.insert(it, Iv{lo, hi, cell});
  }

  void Erase(int r, int cell) {
    auto& v = occ[r];
    for (size_t k = 0; k < v.size(); ++k) {
      if (v[k].cell == cell) {
        v.erase(v.begin() + k);
        return;
      }
    }
  }

  // Nearest legal centre x for `width` in row r, bounded by the die.
  bool NearestFree(int r, double width, double target, double* out) const {
    if (r < rlo || r > rhi) return false;
    const double half = 0.5 * width;
    double best = kInf, bx = 0;
    auto try_gap = [&](double a, double b) {
      const double lo_c = a + half + kGap;
      const double hi_c = b - half - kGap;
      if (lo_c > hi_c) return;
      const double c = std::clamp(target, lo_c, hi_c);
      const double d = std::fabs(c - target);
      if (d < best) {
        best = d;
        bx = c;
      }
    };
    auto it = occ.find(r);
    // Sentinels shifted by kGap so the die edge itself needs no extra margin.
    if (it == occ.end() || it->second.empty()) {
      try_gap(xlo - kGap, xhi + kGap);
    } else {
      const auto& v = it->second;
      try_gap(xlo - kGap, v.front().lo);
      for (size_t k = 0; k + 1 < v.size(); ++k) try_gap(v[k].hi, v[k + 1].lo);
      try_gap(v.back().hi, xhi + kGap);
    }
    if (best >= kInf) return false;
    *out = bx;
    return true;
  }
};

// 1-D optimum of the true objective along x for cell i held at height py.
// Each term is convex in x and piecewise linear with breakpoints at |dx| = |dy|,
// so the minimum is attained at one of those breakpoints.
double BestX(const Problem& P, int i, double py, std::vector<double>& scratch) {
  const auto& A = P.adj[i];
  if (A.empty()) return P.x[i];
  scratch.clear();
  scratch.push_back(P.x[i]);
  for (const Adj& a : A) {
    const double dyv = std::fabs(py - P.y[a.j] + a.ddy);
    const double c = P.x[a.j] - a.ddx;
    scratch.push_back(c - dyv);
    scratch.push_back(c + dyv);
  }
  double best = kInf, bx = P.x[i];
  for (double c : scratch) {
    const double v = CellCost(P, i, c, py);
    if (v < best) {
      best = v;
      bx = c;
    }
  }
  return bx;
}

double BestY(const Problem& P, int i, std::vector<double>& scratch) {
  const auto& A = P.adj[i];
  if (A.empty()) return P.y[i];
  scratch.clear();
  scratch.push_back(P.y[i]);
  for (const Adj& a : A) {
    const double dxv = std::fabs(P.x[i] - P.x[a.j] + a.ddx);
    const double c = P.y[a.j] - a.ddy;
    scratch.push_back(c - dxv);
    scratch.push_back(c + dxv);
  }
  double best = kInf, by = P.y[i];
  for (double c : scratch) {
    const double v = CellCost(P, i, P.x[i], c);
    if (v < best) {
      best = v;
      by = c;
    }
  }
  return by;
}

bool RowLegalize(Problem& P, Rows& R, const Die& die, const std::vector<double>& tx,
                 const std::vector<double>& ty) {
  R.Reset(die);
  if (R.rhi < R.rlo) return false;

  for (int m : P.macros) {
    const double reach = 0.5 * (1.0 + P.h[m]) + kGap;
    const int a = static_cast<int>(std::floor((P.y[m] - reach) / R.pitch));
    const int b = static_cast<int>(std::ceil((P.y[m] + reach) / R.pitch));
    for (int r = std::max(a, R.rlo); r <= std::min(b, R.rhi); ++r) {
      if (std::fabs(R.RowY(r) - P.y[m]) < reach) {
        R.Insert(r, P.x[m] - 0.5 * P.w[m] - kGap, P.x[m] + 0.5 * P.w[m] + kGap, m);
      }
    }
  }

  // Densest-connected cells first: they get first pick of the middle.
  std::vector<int> order = P.stds;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const double ka = (P.adj[a].size() + 1.0) / (P.w[a] * P.h[a]);
    const double kb = (P.adj[b].size() + 1.0) / (P.w[b] * P.h[b]);
    if (ka != kb) return ka > kb;
    return a < b;
  });

  for (int i : order) {
    const int r0 = R.NearestRow(ty[i]);
    double best = kInf;
    int br = -1;
    double bx = 0;
    auto try_row = [&](int r) {
      double px;
      if (!R.NearestFree(r, P.w[i], tx[i], &px)) return;
      const double c = std::fabs(px - tx[i]) + std::fabs(R.RowY(r) - ty[i]);
      if (c < best) {
        best = c;
        br = r;
        bx = px;
      }
    };
    for (int d = -12; d <= 12; ++d) try_row(r0 + d);
    if (br < 0) {
      for (int r = R.rlo; r <= R.rhi; ++r) try_row(r);
    }
    if (br < 0) return false;  // die is too small: caller grows it and retries
    P.x[i] = bx;
    P.y[i] = R.RowY(br);
    R.Insert(br, bx - 0.5 * P.w[i] - kGap, bx + 0.5 * P.w[i] + kGap, i);
  }
  return true;
}

// ---------------------------------------------------------------------------
// stage 4: detailed placement
// ---------------------------------------------------------------------------

void Detailed(Problem& P, Rows& R, int passes, std::mt19937& rng, const TimePoint& deadline) {
  std::vector<int> row_of(P.n, 0);
  for (int i : P.stds) row_of[i] = R.NearestRow(P.y[i]);
  std::vector<int> order = P.stds;
  std::vector<double> scratch;
  scratch.reserve(64);

  for (int pass = 0; pass < passes; ++pass) {
    if (Clock::now() > deadline) break;
    std::shuffle(order.begin(), order.end(), rng);
    int improved = 0;
    for (int i : order) {
      const double cur = CellCost(P, i, P.x[i], P.y[i]);
      const int r_cur = row_of[i];
      R.Erase(r_cur, i);

      const int r_want = R.NearestRow(BestY(P, i, scratch));
      double best = cur;
      int br = r_cur;
      double bx = P.x[i], by = P.y[i];
      auto try_row = [&](int r) {
        if (r < R.rlo || r > R.rhi) return;
        const double yr = R.RowY(r);
        const double t = BestX(P, i, yr, scratch);
        double px;
        if (!R.NearestFree(r, P.w[i], t, &px)) return;
        const double c = CellCost(P, i, px, yr);
        if (c < best - 1e-12) {
          best = c;
          br = r;
          bx = px;
          by = yr;
        }
      };
      for (int d = -3; d <= 3; ++d) {
        try_row(r_cur + d);
        if (r_want != r_cur) try_row(r_want + d);
      }
      P.x[i] = bx;
      P.y[i] = by;
      row_of[i] = br;
      R.Insert(br, bx - 0.5 * P.w[i] - kGap, bx + 0.5 * P.w[i] + kGap, i);
      if (best < cur - 1e-9) ++improved;
    }

    // Swap phase. Relocation cannot help a cell that wants to be inside a full
    // row, so also try exchanging it with an equal-width cell near its target.
    // Equal widths make the exchange legal by construction.
    for (int i : order) {
      const double txi = BestX(P, i, R.RowY(R.NearestRow(BestY(P, i, scratch))), scratch);
      const int rt = R.NearestRow(BestY(P, i, scratch));
      for (int d = -2; d <= 2; ++d) {
        auto it = R.occ.find(rt + d);
        if (it == R.occ.end()) continue;
        const auto& v = it->second;
        if (v.empty()) continue;
        size_t k = static_cast<size_t>(std::lower_bound(v.begin(), v.end(), txi,
                                                        [](const Iv& a, double b) {
                                                          return a.lo < b;
                                                        }) -
                                      v.begin());
        const size_t lo = k > 2 ? k - 2 : 0;
        const size_t hi = std::min(v.size(), k + 3);
        for (size_t t = lo; t < hi; ++t) {
          const int j = v[t].cell;
          if (j == i || P.h[j] > kMacroH || P.w[j] != P.w[i]) continue;
          bool linked = false;
          for (const Adj& a : P.adj[i]) {
            if (a.j == j) {
              linked = true;
              break;
            }
          }
          if (linked) continue;  // delta would need special-casing; rare, skip
          const double old_c = CellCost(P, i, P.x[i], P.y[i]) + CellCost(P, j, P.x[j], P.y[j]);
          const double new_c = CellCost(P, i, P.x[j], P.y[j]) + CellCost(P, j, P.x[i], P.y[i]);
          if (new_c < old_c - 1e-9) {
            const int ri = row_of[i], rj = row_of[j];
            R.Erase(ri, i);
            R.Erase(rj, j);
            std::swap(P.x[i], P.x[j]);
            std::swap(P.y[i], P.y[j]);
            std::swap(row_of[i], row_of[j]);
            R.Insert(row_of[i], P.x[i] - 0.5 * P.w[i] - kGap, P.x[i] + 0.5 * P.w[i] + kGap, i);
            R.Insert(row_of[j], P.x[j] - 0.5 * P.w[j] - kGap, P.x[j] + 0.5 * P.w[j] + kGap, j);
            ++improved;
            break;
          }
        }
      }
    }
    if (improved == 0) break;
  }
}

// ---------------------------------------------------------------------------
// exact legality check, mirroring calculate_cells_with_overlaps()
// ---------------------------------------------------------------------------

int CountOverlappingCells(const Problem& P) {
  const int n = P.n;
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return P.x[a] - 0.5 * P.w[a] < P.x[b] - 0.5 * P.w[b];
  });
  std::vector<char> bad(n, 0);
  std::vector<int> active;
  for (int k = 0; k < n; ++k) {
    const int i = idx[k];
    const double lo_i = P.x[i] - 0.5 * P.w[i];
    size_t out = 0;
    for (size_t t = 0; t < active.size(); ++t) {
      const int j = active[t];
      if (P.x[j] + 0.5 * P.w[j] <= lo_i) continue;  // can never overlap again
      active[out++] = j;
      const double ox = 0.5 * (P.w[i] + P.w[j]) - std::fabs(P.x[i] - P.x[j]);
      const double oy = 0.5 * (P.h[i] + P.h[j]) - std::fabs(P.y[i] - P.y[j]);
      if (ox > 0.0 && oy > 0.0) {
        bad[i] = 1;
        bad[j] = 1;
      }
    }
    active.resize(out);
    active.push_back(i);
  }
  int c = 0;
  for (int i = 0; i < n; ++i) c += bad[i];
  return c;
}

void RoundToFloat32(Problem& P) {
  for (int i = 0; i < P.n; ++i) {
    P.x[i] = static_cast<double>(static_cast<float>(P.x[i]));
    P.y[i] = static_cast<double>(static_cast<float>(P.y[i]));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

extern "C" int partcl_place(int n, const double* w_in, const double* h_in, double* x_io,
                            double* y_io, int n_pins, const int* pin_cell, const double* pin_ox,
                            const double* pin_oy, int n_edges, const int* ea_in, const int* eb_in,
                            double budget_s, unsigned seed, int verbose) {
  const auto t_start = Clock::now();
  const auto deadline =
      t_start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget_s));

  Problem P;
  P.n = n;
  P.w.assign(w_in, w_in + n);
  P.h.assign(h_in, h_in + n);
  P.x.assign(n, 0.0);
  P.y.assign(n, 0.0);
  P.adj.assign(n, {});
  P.total_edges = n_edges;
  for (int i = 0; i < n; ++i) {
    P.area += P.w[i] * P.h[i];
    (P.h[i] > kMacroH ? P.macros : P.stds).push_back(i);
  }
  P.side = std::sqrt(std::max(P.area, 1e-12));

  for (int e = 0; e < n_edges; ++e) {
    const int pa = ea_in[e], pb = eb_in[e];
    const int ci = pin_cell[pa], cj = pin_cell[pb];
    const double ddx = pin_ox[pa] - pin_ox[pb];
    const double ddy = pin_oy[pa] - pin_oy[pb];
    if (ci == cj) {
      P.const_cost += EdgeCost(ddx, ddy);
      continue;
    }
    P.ea.push_back(ci);
    P.eb.push_back(cj);
    P.edx.push_back(ddx);
    P.edy.push_back(ddy);
    P.adj[ci].push_back(Adj{cj, ddx, ddy});
    P.adj[cj].push_back(Adj{ci, -ddx, -ddy});
  }
  (void)n_pins;

  std::mt19937 rng(seed);

  // One unconstrained global placement, reused as the starting point for every
  // die/macro configuration below.
  {
    std::uniform_real_distribution<double> U(-0.6 * P.side, 0.6 * P.side);
    for (int i = 0; i < n; ++i) {
      P.x[i] = U(rng);
      P.y[i] = U(rng);
    }
    const std::vector<char> all(n, 1);
    GlobalPlace(P, all, n > 1000 ? 300 : 600, nullptr);
  }
  const std::vector<double> gx0 = P.x, gy0 = P.y;

  std::vector<char> movable(n, 1);
  for (int m : P.macros) movable[m] = 0;

  // Candidate schedule: die aspect ratio x macro arrangement.
  const double aspects[] = {1.0,  0.85, 1.18, 0.72, 1.4,  0.6, 1.667,
                            0.5,  2.0,  0.4,  2.5,  0.32, 3.1};
  const int n_aspect = 13;
  // 0        = macros legalised from the unconstrained global placement
  // 1..8     = shelf-packed into an edge band (4 corners x 2 orientations)
  // 9..12    = perturb the best macro arrangement found so far, shrinking sigma.
  //            Shelf packing only produces lattice-aligned macro offsets; these
  //            modes search the off-lattice offsets, which is where the small
  //            macro-dominated designs have their remaining slack.
  const int n_mode = 14;

  std::vector<int> ffdh = P.macros;
  std::sort(ffdh.begin(), ffdh.end(), [&](int a, int b) { return P.h[a] > P.h[b]; });
  std::vector<int> ffdw = P.macros;
  std::sort(ffdw.begin(), ffdw.end(), [&](int a, int b) { return P.w[a] > P.w[b]; });

  std::vector<double> best_x, best_y;
  double best_cost = kInf;
  int cand = 0, legal_cands = 0, best_ai = 0;
  double util_hint = 0.95;
  const double per_cand = std::max(0.02, budget_s / 8.0);

  for (int round = 0; ; ++round) {
    for (int ai_i = 0; ai_i < n_aspect; ++ai_i) {
      for (int mode_i = 0; mode_i < n_mode; ++mode_i) {
        // Never leave without at least one legal placement, budget or not.
        if (Clock::now() > deadline && legal_cands > 0) goto done;

        // Round 0 is a deterministic sweep of every die aspect against every
        // constructive macro arrangement. After that, stop re-scanning aspects
        // that already lost and hill-climb the macro offsets at the winning
        // aspect instead, annealing sigma down through modes 9..12.
        int ai = ai_i, mode = mode_i;
        if (round == 0) {
          if (mode > 8) continue;
        } else {
          ai = best_ai;
          mode = 9 + (mode_i % 5);
        }
        ++cand;

        Die die;
        Rows R;
        std::vector<double> tx(n), ty(n);
        std::vector<int> macro_order = ((mode - 1) / 4 == 1) ? ffdw : ffdh;
        if (mode > 0 && mode <= 8 && round > 0) {
          std::shuffle(macro_order.begin(), macro_order.end(), rng);
        }
        // Perturbation modes need a best arrangement to perturb.
        std::vector<double> ptx = gx0, pty = gy0;
        if (mode >= 9) {
          if (best_x.empty()) continue;
          if (mode == 13) {
            // Directed move: aim each macro at the 1-D optimum of the real
            // objective given the current best placement. Random perturbation
            // alone barely moves a macro usefully once the layout is decent.
            P.x = best_x;
            P.y = best_y;
            std::vector<double> sc;
            sc.reserve(512);
            for (int m : P.macros) {
              ptx[m] = BestX(P, m, P.y[m], sc);
              pty[m] = BestY(P, m, sc);
            }
          } else {
            const double sig = P.side * (mode == 9 ? 0.40 : mode == 10 ? 0.15
                                                      : mode == 11 ? 0.05 : 0.015);
            std::normal_distribution<double> N(0.0, sig);
            for (int m : P.macros) {
              ptx[m] = best_x[m] + N(rng);
              pty[m] = best_y[m] + N(rng);
            }
          }
        }

        // Feasibility probe at a given utilisation: place the macros, then try to
        // row-legalise every standard cell. Cheap, so the tightest feasible die
        // can be found by search rather than guessed.
        auto probe = [&](double util) {
          const double A = P.area / util;
          die.W = std::sqrt(A * aspects[ai]);
          die.H = A / die.W;
          bool macros_ok;
          if (mode == 0) {
            macros_ok = LegalizeMacrosGreedy(P, gx0, gy0, die);
          } else if (mode <= 8) {
            const int corner = (mode - 1) % 4;
            const bool vertical = ((mode - 1) / 4 == 1);
            macros_ok = ShelfPackMacros(P, die, macro_order, corner, vertical);
          } else {
            macros_ok = LegalizeMacrosGreedy(P, ptx, pty, die);
          }
          if (!macros_ok) return false;
          for (int i : P.stds) {
            tx[i] = std::clamp(gx0[i], -0.5 * die.W + 0.5 * P.w[i], 0.5 * die.W - 0.5 * P.w[i]);
            ty[i] = std::clamp(gy0[i], -0.5 * die.H + 0.5 * P.h[i], 0.5 * die.H - 0.5 * P.h[i]);
          }
          return RowLegalize(P, R, die, tx, ty);
        };

        // Walk down from a hint until it fits, then bisect to tighten. util_hint
        // carries across candidates because the feasible utilisation barely
        // moves between die shapes, which keeps this to a couple of probes.
        double hi_fail = std::min(1.0, util_hint * 1.04);
        double lo_ok = -1.0;
        {
          double u = hi_fail;
          bool ok = false;
          for (int s = 0; s < 30; ++s) {
            if (probe(u)) {
              ok = true;
              lo_ok = u;
              break;
            }
            hi_fail = u;
            u *= 0.985;
          }
          if (!ok) continue;
        }
        for (int b = 0; b < 5; ++b) {
          const double mid = 0.5 * (lo_ok + hi_fail);
          if (mid <= lo_ok * (1.0 + 1e-4)) break;
          if (probe(mid)) {
            lo_ok = mid;
          } else {
            hi_fail = mid;
          }
        }
        if (!probe(lo_ok)) continue;  // restore the best feasible die
        util_hint = lo_ok;
        ++legal_cands;

        // Refine the standard-cell targets against this die and macro plan,
        // warm-started from the legal placement we just built.
        GlobalPlace(P, movable, n > 1000 ? 90 : 180, &die);
        for (int i : P.stds) {
          tx[i] = P.x[i];
          ty[i] = P.y[i];
        }
        if (!RowLegalize(P, R, die, tx, ty)) {
          for (int i : P.stds) {
            tx[i] = std::clamp(gx0[i], -0.5 * die.W + 0.5 * P.w[i], 0.5 * die.W - 0.5 * P.w[i]);
            ty[i] = std::clamp(gy0[i], -0.5 * die.H + 0.5 * P.h[i], 0.5 * die.H - 0.5 * P.h[i]);
          }
          if (!RowLegalize(P, R, die, tx, ty)) continue;
        }

        const auto cand_deadline =
            std::min(deadline, Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                                  std::chrono::duration<double>(per_cand)));
        Detailed(P, R, 60, rng, cand_deadline);
        // The die is our own construction, not part of the problem: there is no
        // fixed outline in this challenge. It exists only to force a dense
        // packing. Now drop it and let detailed placement keep going without
        // bounds. Every move must strictly reduce the objective, so releasing
        // the constraint can only help.
        R.xlo = -1e9;
        R.xhi = 1e9;
        R.rlo = -1000000;
        R.rhi = 1000000;
        Detailed(P, R, 30, rng, cand_deadline);
        RoundToFloat32(P);
        const int bad = CountOverlappingCells(P);
        const double cost = NormalizedWL(P) + (bad ? 1e6 : 0.0);
        if (cost < best_cost) {
          best_cost = cost;
          best_x = P.x;
          best_y = P.y;
          best_ai = ai;
          if (verbose) {
            std::fprintf(stderr, "    cand %d (aspect %.2f mode %d): wl=%.4f bad=%d\n", cand,
                         aspects[ai], mode, NormalizedWL(P), bad);
          }
        }
      }
    }
    if (Clock::now() > deadline) break;
    if (P.macros.size() <= 1 && round >= 1) break;  // nothing left to permute
    if (round > 40) break;
  }
done:
  if (best_x.empty()) {  // should not happen; keep whatever we have
    best_x = P.x;
    best_y = P.y;
  }
  P.x = best_x;
  P.y = best_y;
  const int bad = CountOverlappingCells(P);
  for (int i = 0; i < n; ++i) {
    x_io[i] = P.x[i];
    y_io[i] = P.y[i];
  }
  if (verbose) {
    const double s = std::chrono::duration<double>(Clock::now() - t_start).count();
    std::fprintf(stderr, "  cands=%d legal=%d wl=%.4f overlap_cells=%d %.2fs\n", cand, legal_cands,
                 NormalizedWL(P), bad, s);
  }
  return bad;
}
