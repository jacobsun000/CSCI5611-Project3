#pragma once

#include <cmath>
#include <limits>
#include <random>

namespace Math {

using std::abs;
using std::max;
using std::min;

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.1415926535897932385;
constexpr double EPSILON = 1e-6;

inline constexpr double relu(double x) { return x > 0 ? x : 0; }

inline constexpr double deg2rad(double degrees) { return degrees * PI / 180.0; }

inline constexpr double rad2deg(double radians) { return radians / PI * 180.0; }

// Returns a random real in [0,1).
inline double random_double() {
  thread_local std::minstd_rand generator(std::random_device{}());
  thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);

  return distribution(generator);
}

// Returns a random real in [min,max).
inline double random_double(double min, double max) {
  static std::mt19937 generator;
  std::uniform_real_distribution<double> distribution(min, max);
  return distribution(generator);
}

struct wangshash {
  uint32_t a;

  explicit wangshash() : a(static_cast<uint32_t>(std::random_device{}())) {}

  using result_type = uint32_t;

  constexpr uint32_t operator()() noexcept {
    uint32_t x = a;
    x = (x ^ 61) ^ (x >> 16);
    x *= 9;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2d;
    x = x ^ (x >> 15);
    return a = x;
  }

  static constexpr uint32_t min() noexcept { return 0; }

  static constexpr uint32_t max() noexcept { return UINT32_MAX; }
};

static wangshash generator{};

// Returns a random real in [min,max).
inline double random_float(float min, float max) {
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(generator);
}

// Clamp x to a range of [min, max]
inline constexpr double clamp(double x, double min, double max) {
  if (x < min)
    return min;
  if (x > max)
    return max;
  return x;
}

// Return true if x is in range [min, max]
inline constexpr bool within_range(double x, double min, double max) {
  return x >= min && x <= max;
}

}; // namespace Math
