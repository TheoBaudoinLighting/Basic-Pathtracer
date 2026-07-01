// -----------------------------------------------------------------------------
// Basic Pathtracer
//
// This code is owned by Théo Baudoin-Malnoë
// It is provided as is, without warranty of any kind
// The author does not guarantee that it works and is not liable for any use
// of this code
// -----------------------------------------------------------------------------
//
// Standalone demo renderer: Cornell box with a mirror and a glass sphere
// Path tracer with next event estimation on the area light, specular and
// dielectric materials, Russian roulette, and PPM output
//
// Usage: cornell_spheres [W H spp out.ppm]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Small 3D vector type with explicit math so the renderer can be read without
// depending on an external linear algebra library
struct V3 {
  double x = 0.0, y = 0.0, z = 0.0;
};

V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
V3 mulv(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

V3 cross(V3 a, V3 b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

V3 norm(V3 a) {
  const double l = std::sqrt(dot(a, a));
  return {a.x / l, a.y / l, a.z / l};
}

enum class Mat { kDiffuse, kMirror, kGlass, kLight };

struct HitInfo {
  double t = kInf;         // Distance from ray origin to closest hit
  V3 p{}, n{};             // Hit position and surface normal
  V3 albedo{};             // Surface color used by non-emissive materials
  Mat mat = Mat::kDiffuse; // Material at the hit point
};

struct Rect {
  int axis; // Constant axis: 0 = x, 1 = y, 2 = z
  // Plane position and bounds on the two axes that are not constant
  double pos, lo1, hi1, lo2, hi2;
  V3 n, albedo; // Fixed normal and material color
  Mat mat;
};

struct Sphere {
  V3 c;
  double r;
  V3 albedo;
  Mat mat;
};

double comp(V3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }
int oa1(int a) { return a == 0 ? 1 : 0; }
int oa2(int a) { return a == 2 ? 1 : 2; }

struct SceneS {
  std::vector<Rect> rects;
  std::vector<Sphere> spheres;
  double ly, lu0, lu1, lv0, lv1; // Area light plane and 2D bounds
  V3 le;                         // Emitted radiance of the area light
};

SceneS MakeScene() {
  SceneS s;
  const V3 white{0.73, 0.73, 0.73};
  const V3 red{0.63, 0.065, 0.05};
  const V3 green{0.14, 0.45, 0.091};

  // Cornell box: floor, ceiling, back wall, left wall, right wall, then light
  s.rects.push_back(
      {1, 0.0, 0.0, 556.0, 0.0, 559.2, {0, 1, 0}, white, Mat::kDiffuse});
  s.rects.push_back(
      {1, 548.8, 0.0, 556.0, 0.0, 559.2, {0, -1, 0}, white, Mat::kDiffuse});
  s.rects.push_back(
      {2, 559.2, 0.0, 556.0, 0.0, 548.8, {0, 0, -1}, white, Mat::kDiffuse});
  s.rects.push_back(
      {0, 0.0, 0.0, 548.8, 0.0, 559.2, {1, 0, 0}, red, Mat::kDiffuse});
  s.rects.push_back(
      {0, 556.0, 0.0, 548.8, 0.0, 559.2, {-1, 0, 0}, green, Mat::kDiffuse});
  s.rects.push_back({1,
                     548.0,
                     213.0,
                     343.0,
                     227.0,
                     332.0,
                     {0, -1, 0},
                     {0, 0, 0},
                     Mat::kLight});

  // Two centered spheres with different light transport behavior
  s.spheres.push_back(
      {{168.0, 90.0, 145.0}, 90.0, {0.98, 0.98, 0.98}, Mat::kMirror});
  s.spheres.push_back(
      {{390.0, 90.0, 360.0}, 90.0, {0.99, 0.99, 0.99}, Mat::kGlass});

  s.ly = 548.0;
  s.lu0 = 213.0;
  s.lu1 = 343.0;
  s.lv0 = 227.0;
  s.lv1 = 332.0;
  s.le = {15.0, 15.0, 15.0};
  return s;
}

void HitRect(const Rect &r, V3 o, V3 d, HitInfo &h) {
  // Intersect a ray with an axis-aligned rectangle, where the plane is constant
  // on r.axis and the hit point is checked against the two remaining axes
  const double dd = comp(d, r.axis);
  if (std::fabs(dd) < 1e-12)
    return;

  const double t = (r.pos - comp(o, r.axis)) / dd;
  if (t < 1e-4 || t >= h.t)
    return;

  const V3 p = o + d * t;
  const double c1 = comp(p, oa1(r.axis));
  const double c2 = comp(p, oa2(r.axis));
  if (c1 < r.lo1 || c1 > r.hi1 || c2 < r.lo2 || c2 > r.hi2)
    return;

  h.t = t;
  h.p = p;
  h.n = r.n;
  h.albedo = r.albedo;
  h.mat = r.mat;
}

void HitSphere(const Sphere &s, V3 o, V3 d, HitInfo &h) {
  // For normalized ray directions, the quadratic term is one and the equation
  // is written in the compact form t^2 + 2*b*t + c = 0
  const V3 oc = o - s.c;
  const double b = dot(oc, d);
  const double c = dot(oc, oc) - s.r * s.r;
  const double disc = b * b - c;
  if (disc <= 0.0)
    return;

  const double sq = std::sqrt(disc);
  double t = -b - sq;
  if (t < 1e-4)
    t = -b + sq;
  if (t < 1e-4 || t >= h.t)
    return;

  h.t = t;
  h.p = o + d * t;
  h.n = norm(h.p - s.c);
  h.albedo = s.albedo;
  h.mat = s.mat;
}

HitInfo TraceS(const SceneS &s, V3 o, V3 d) {
  HitInfo h;

  // Every primitive tries to improve the closest hit stored in h
  for (const Rect &r : s.rects)
    HitRect(r, o, d, h);
  for (const Sphere &sp : s.spheres)
    HitSphere(sp, o, d, h);

  return h;
}

struct Rng {
  std::mt19937_64 g;
  std::uniform_real_distribution<double> u{0.0, 1.0};
  double operator()() { return u(g); }
};

V3 SampleHemi(V3 n, Rng &r) {
  // Cosine-weighted hemisphere sampling, diffuse reflection is stronger near
  // the surface normal, and this distribution matches that shape
  const double u1 = r();
  const double u2 = r();
  const double rad = std::sqrt(u1);
  const double th = 2.0 * M_PI * u2;

  // Build an orthonormal basis around n, then sample in that local frame
  const V3 a = std::fabs(n.x) > 0.5 ? V3{0, 1, 0} : V3{1, 0, 0};
  const V3 t = norm(cross(a, n));
  const V3 b = cross(n, t);

  return norm(t * (rad * std::cos(th)) + b * (rad * std::sin(th)) +
              n * std::sqrt(std::max(0.0, 1.0 - u1)));
}

bool Shadowed(const SceneS &s, V3 a, V3 b) {
  // Trace toward the sampled light point, a hit before t=1 means something is
  // between the shading point and the light
  const V3 d = b - a;
  HitInfo h = TraceS(s, a, d);
  return h.t < 1.0 - 1e-4;
}

V3 Nee(const SceneS &s, V3 x, V3 n, V3 albedo, Rng &rng) {
  // Next event estimation: sample the area light directly from a diffuse hit
  // This reduces noise compared with waiting for random bounces to find it
  const double lu = s.lu0 + (s.lu1 - s.lu0) * rng();
  const double lv = s.lv0 + (s.lv1 - s.lv0) * rng();
  const V3 y{lu, s.ly, lv};

  const V3 dv = y - x;
  const double r2 = dot(dv, dv);
  const double c1 = dot(dv, n);
  const double c2 = dv.y;
  if (c1 <= 0.0 || c2 <= 0.0)
    return {0, 0, 0};
  if (Shadowed(s, x + n * 1e-3, y))
    return {0, 0, 0};

  const double area = (s.lu1 - s.lu0) * (s.lv1 - s.lv0);
  const double w = c1 * c2 / (r2 * r2) * area / M_PI;
  return mulv(albedo, s.le) * w;
}

V3 Radiance(const SceneS &s, V3 o, V3 d, Rng &rng) {
  V3 acc{0, 0, 0};  // Accumulated radiance returned by the path
  V3 beta{1, 1, 1}; // Path throughput: remaining energy carried by the ray

  // Direct light hits are only added after camera/specular paths, diffuse hits
  // use Nee() instead, which avoids double-counting the area light
  bool specular = true;

  for (int depth = 0; depth < 24; ++depth) {
    const HitInfo h = TraceS(s, o, d);
    if (h.t == kInf)
      break;

    if (h.mat == Mat::kLight) {
      if (specular && dot(d, h.n) < 0.0)
        acc = acc + mulv(beta, s.le);
      break;
    }

    if (h.mat == Mat::kDiffuse) {
      acc = acc + mulv(beta, Nee(s, h.p, h.n, h.albedo, rng));
      beta = mulv(beta, h.albedo);
      specular = false;
      d = SampleHemi(h.n, rng);
      o = h.p + h.n * 1e-3;
    } else if (h.mat == Mat::kMirror) {
      beta = mulv(beta, h.albedo);
      specular = true;
      d = norm(d - h.n * (2.0 * dot(d, h.n)));
      o = h.p + d * 1e-3;
    } else {
      // Glass
      // Snell refraction with Schlick's approximation for the reflection
      // chance
      const bool into = dot(d, h.n) < 0.0;
      const V3 nl = into ? h.n : h.n * -1.0;
      const double eta = into ? 1.0 / 1.5 : 1.5;
      const double c = -dot(d, nl);
      const double k = 1.0 - eta * eta * (1.0 - c * c);
      specular = true;

      if (k < 0.0) {
        // Total internal reflection
        d = norm(d + nl * (2.0 * c));
        o = h.p + d * 1e-3;
      } else {
        const double ct = std::sqrt(k);
        const double r0 =
            ((1.5 - 1.0) * (1.5 - 1.0)) / ((1.5 + 1.0) * (1.5 + 1.0));
        const double cx = into ? c : ct;
        const double fr = r0 + (1.0 - r0) * std::pow(1.0 - cx, 5.0);

        if (rng() < fr) {
          d = norm(d + nl * (2.0 * c));
        } else {
          beta = mulv(beta, h.albedo);
          d = norm(d * eta + nl * (eta * c - ct));
        }
        o = h.p + d * 1e-3;
      }
    }

    if (depth >= 4) {
      // Russian roulette stops low-energy paths while keeping the estimator
      // unbiased by boosting paths that survive
      const double p =
          std::clamp(std::max({beta.x, beta.y, beta.z}), 0.05, 0.95);
      if (rng() > p)
        break;
      beta = beta * (1.0 / p);
    }
  }
  return acc;
}

} // namespace

int main(int argc, char **argv) {
  const int kw = argc >= 3 ? std::atoi(argv[1]) : 768;
  const int kh = argc >= 3 ? std::atoi(argv[2]) : 768;
  const int spp = argc >= 4 ? std::atoi(argv[3]) : 4096;
  const char *out = argc >= 5 ? argv[4] : "cornell_spheres.ppm";

  const SceneS scene = MakeScene();
  const V3 eye{278.0, 273.0, -800.0};
  const double tan_half = std::tan(0.5 * 39.3076 * M_PI / 180.0);

  std::vector<V3> img(static_cast<std::size_t>(kw) * kh);

  // Rows are distributed dynamically so all threads keep working even if some
  // rows happen to take longer than others
  std::atomic<int> next_row{0};
  const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;

  for (unsigned ti = 0; ti < nt; ++ti) {
    pool.emplace_back([&] {
      for (;;) {
        const int y = next_row.fetch_add(1);
        if (y >= kh)
          break;
        Rng rng;
        rng.g.seed(4242u + static_cast<unsigned>(y));
        for (int x = 0; x < kw; ++x) {
          V3 c{0, 0, 0};

          // Jitter each camera ray inside the pixel and average the samples
          for (int k = 0; k < spp; ++k) {
            const double sx = (2.0 * (x + rng()) / kw - 1.0) * tan_half;
            const double sy = (1.0 - 2.0 * (y + rng()) / kh) * tan_half;
            c = c + Radiance(scene, eye, norm({sx, sy, 1.0}), rng);
          }
          img[static_cast<std::size_t>(y) * kw + x] = c * (1.0 / spp);
        }
      }
    });
  }
  for (std::thread &t : pool)
    t.join();

  // Binary PPM (P6): tiny format, easy to write, no image library required
  std::FILE *f = std::fopen(out, "wb");
  std::fprintf(f, "P6\n%d %d\n255\n", kw, kh);

  for (const V3 &c : img) {
    const auto q = [](double v) {
      // Clamp to display range and apply a simple gamma curve before
      // quantizing
      const double g = std::pow(std::clamp(v, 0.0, 1.0), 1.0 / 2.2);
      return static_cast<unsigned char>(std::lround(255.0 * g));
    };
    const unsigned char px[3] = {q(c.x), q(c.y), q(c.z)};
    std::fwrite(px, 1, 3, f);
  }
  std::fclose(f);
  std::printf("wrote %s (%dx%d, %d spp)\n", out, kw, kh, spp);
  return 0;
}
