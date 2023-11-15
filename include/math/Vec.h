#pragma once
#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <type_traits>

#include "MathUtils.h"

template <typename T, size_t N> struct Vec;

template <typename T, size_t Rows, size_t Cols> struct Mat;

template <size_t N> using VecNf = Vec<float, N>;
using Vecf = VecNf<0>;
using Vec1f = VecNf<1>;
using Vec2f = VecNf<2>;
using Vec3f = VecNf<3>;
using Vec4f = VecNf<4>;
using Pointf = Vecf;
using Point1f = Vec1f;
using Point2f = Vec2f;
using Point3f = Vec3f;
using Point4f = Vec4f;

template <size_t N> using VecNd = Vec<double, N>;
using Vecd = VecNd<0>;
using Vec1d = VecNd<1>;
using Vec2d = VecNd<2>;
using Vec3d = VecNd<3>;
using Vec4d = VecNd<4>;
using Pointd = Vecd;
using Point1d = Vec1d;
using Point2d = Vec2d;
using Point3d = Vec3d;
using Point4d = Vec4d;

template <typename T, size_t N = 0> struct Vec {
  static constexpr size_t MaxStackSize = 4;
  static constexpr size_t Dynamic = 0;
  static constexpr bool UseStack = (N <= MaxStackSize && N != Dynamic);
  using StorageType = typename std::conditional<UseStack, std::array<T, N>,
                                                std::vector<T>>::type;

  StorageType data;

  constexpr Vec() = default;

  // Fill length N vector with value of type T
  constexpr explicit Vec(T value) { data.fill(value); }

  explicit Vec(size_t n, T val) {
    static_assert(!UseStack, "Cannot resize stack allocated vector.");
    data.resize(n, val);
  }

  constexpr Vec(const T (&arr)[N]) {
    std::copy(std::begin(arr), std::end(arr), std::begin(data));
  }

  // Fill length N vector with a list
  constexpr Vec(std::initializer_list<T> values) {
    if (values.size() != N && N != 0) {
      throw std::runtime_error("Invalid initializer list length.");
    }
    std::copy(values.begin(), values.end(), data.begin());
  }

  Vec &operator=(const Vec &other) {
    if constexpr (UseStack) {
      data = other.data;
    } else {
      data.resize(other.size());
      std::copy(other.data.begin(), other.data.end(), data.begin());
    }
    return *this;
  }

  [[nodiscard]] constexpr size_t size() const {
    if constexpr (UseStack) {
      return N; // For stack allocated, Rows is a compile-time constant
    } else {
      return data.size();
    }
  }

  static constexpr Vec<T, N> one() { return Vec<T, N>(1); }

  static constexpr Vec<T, N> zero() { return Vec<T, N>(0); }

  void resize(size_t n) {
    if constexpr (UseStack) {
      throw std::runtime_error("Cannot resize stack allocated vector.");
    } else {
      data.resize(n);
    }
  }

  T &operator[](size_t index) { return data[index]; }

  const T &operator[](size_t index) const { return data[index]; }

  template <class U = T>
  constexpr typename std::enable_if<(N >= 1), U &>::type x() {
    return data[0];
  }

  template <class U = T>
  constexpr typename std::enable_if<(N >= 2), U &>::type y() {
    return data[1];
  }

  template <class U = T>
  constexpr typename std::enable_if<(N >= 3), U &>::type z() {
    return data[2];
  }

  // Magnitude of vector
  T length() const { return sqrt(length_squared()); }

  // Square length of vector
  T length_squared() const {
    T result = 0;
    for (const auto &value : data) {
      result += value * value;
    }
    return result;
  }

  // Return a new unit vector of current vector
  Vec<T, N> normalize() const { return (*this) / length(); }

  // Normalize current vector
  Vec<T, N> &normalized() {
    *this /= length();
    return *this;
  }

  Vec<T, N> to_length(T len) const { return normalize() * len; }

  T sum() const {
    return fold<T>([](T acc, T val) { return acc + val; }, 0.0);
  }

  T abs_sum() const {
    return fold<T>([](T acc, T val) { return acc + fabs(val); }, 0.0);
  }

  Vec<T, N> clamp() const {
    return map<T>([](T val) { return Math::clamp(val, 0.0, 1.0); });
  }

  // Return true if the vector is close to zero in all dimensions.
  bool near_zero() const {
    return fold(
        [](bool acc, T val) { return acc && (fabs(val) < Math::EPSILON); },
        true);
  }

  // Dot product of vector with another vector
  T dot(const Vec<T, N> &other) const {
    if (size() != other.size()) {
      throw std::runtime_error("Cannot dot vectors of different sizes.");
    }
    T result = 0;
    for (size_t i = 0; i < size(); ++i) {
      result += data[i] * other[i];
    }
    return result;
  }

  T cross(const Vec<T, 2> &other) const {
    if (size() != 2) {
      throw std::runtime_error("Cross product is only defined for 2D vectors.");
    }
    return (*this)[0] * other[1] - (*this)[1] * other[0];
  }

  // Cross product of vector with another vector.
  // Only availiable in Vec<T, 3>
  Vec cross(const Vec<T, 3> &other) const {
    if (size() != 3) {
      throw std::runtime_error("Cross product is only defined for 3D vectors.");
    }
    Vec<T, 3> result;
    result[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
    result[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
    result[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
    return result;
  }

  template <size_t M = N>
  Mat<T, M, N> outer_product(const Vec<T, M> &other) const {
    Mat<T, M, N> result;
    if constexpr (!UseStack || !other.UseStack) {
      result.resize(M, N);
    }
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        result[i][j] = other[i] * data[j];
      }
    }
    return result;
  }

  // Map each component of a vector to a new vector using func
  template <typename U = T> Vec map(std::function<U(T)> func) const {
    Vec<U, N> result;
    if constexpr (!UseStack) {
      result.resize(size());
    }
    for (size_t i = 0; i < size(); i++) {
      result[i] = func(data[i]);
    }
    return result;
  }

  // Map each component of a vector to a new vector using func
  template <typename U = T> Vec map(std::function<U(T, size_t)> func) const {
    Vec<U, N> result;
    if constexpr (!UseStack) {
      result.resize(size());
    }
    for (size_t i = 0; i < size(); i++) {
      result[i] = func(data[i], i);
    }
    return result;
  }

  // Apply func to each component of a vector
  Vec &apply(std::function<void(T &)> func) {
    for (T &value : data) {
      func(value);
    }
    return *this;
  }

  // Apply func to each component of a vector
  Vec &apply(std::function<void(T &, size_t)> func) {
    for (size_t i = 0; i < size(); i++) {
      func(data[i], i);
    }
    return *this;
  }

  // Fold a vector to a single value using func
  template <typename U = T>
  U fold(std::function<U(U, T)> func, U initial) const {
    U result = initial;
    for (size_t i = 0; i < size(); i++) {
      result = func(result, data[i]);
    }
    return result;
  }

#define _vec_unary_op(op)                                                      \
  Vec operator op() const {                                                    \
    return map<T>([](T val) { return op val; });                               \
  }

  _vec_unary_op(+);
  _vec_unary_op(-);
  _vec_unary_op(~);
  _vec_unary_op(!);
#undef _vec_unary_op

#define _vec_op_scalar(op)                                                     \
  friend Vec operator op(const T &scalar, const Vec<T, N> &vec) {              \
    return vec op scalar;                                                      \
  }                                                                            \
  Vec operator op(const T &scalar) const {                                     \
    return map<T>([&](T val, size_t i) { return val op scalar; });             \
  }                                                                            \
  Vec &operator op##=(const T &scalar) {                                       \
    return apply([&](T &val, size_t i) { val = val op scalar; });              \
  }

  _vec_op_scalar(+);
  _vec_op_scalar(-);
  _vec_op_scalar(*);
  _vec_op_scalar(/);
  _vec_op_scalar(%);
#undef _vec_op_scalar

#define _vec_op_vec(op)                                                        \
  Vec operator op(const Vec<T, N> &other) const {                              \
    return map<T>([&](T val, size_t i) { return val op other[i]; });           \
  }                                                                            \
  Vec &operator op##=(const Vec<T, N> &other) {                                \
    return apply([&](T &val, size_t i) { val = val op other[i]; });            \
  }

  _vec_op_vec(+);
  _vec_op_vec(-);
  _vec_op_vec(*);
  _vec_op_vec(/);
  _vec_op_vec(%);
#undef _vec_op_vec

  bool operator==(const Vec<T, N> &other) const {
    for (size_t i = 0; i < size(); i++) {
      if (this->data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Vec<T, N> &other) const { return !(*this == other); }

  friend std::ostream &operator<<(std::ostream &os, const Vec<T, N> &vec) {
    os << "[ ";
    os << vec[0];
    for (std::size_t i = 1; i < vec.size(); i++) {
      os << ", " << vec[i];
    }
    os << " ]";
    return os;
  }

  friend std::istream &operator>>(std::istream &is, Vec<T, N> &vec) {
    is.ignore();
    for (size_t i = 0; i < vec.size(); ++i) {
      is >> vec[i];
      if (i != vec.size() - 1) {
        is.ignore(2);
      }
    }
    is.ignore(1, '\n');
    if (is.fail()) {
      throw std::runtime_error("Failed to read Vec");
    }
    return is;
  }

  static Vec<T, N> random(double min, double max) {
    Vec<T, N> result;
    for (size_t i = 0; i < N; i++) {
      result[i] = Math::random_double(min, max);
    }
    return result;
  }

  static Vec<T, N> random() {
    Vec<T, N> result;
    for (size_t i = 0; i < N; i++) {
      result[i] = Math::random_double();
    }
    return result;
  }

  static Vec<double, N> random_unit_vector() {
    Vec<double, N> result;
    do {
      result = Vec<double, N>::random(-1, 1);
    } while (result.length() >= 1.0);
    return result.unit_vector();
  }

  static Vec<double, N> random_in_hemisphere(const Vec<double, N> &normal) {
    Vec<double, N> result = random_unit_vector();
    if (result.dot(normal) < 0) {
      return -result;
    }
    return result;
  }
};
template <typename T, typename... Ts>
Vec(T, Ts...) -> Vec<T, 1 + sizeof...(Ts)>;
