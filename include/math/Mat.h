#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <execution>
#include <initializer_list>
#include <iostream>

#include "Vec.h"

template <typename T, size_t Rows, size_t Cols>
struct Mat;

template <size_t Rows, size_t Cols>
using MatNf = Mat<float, Rows, Cols>;
using Matf = MatNf<0, 0>;
using Mat11f = MatNf<1, 1>;
using Mat12f = MatNf<1, 2>;
using Mat13f = MatNf<1, 3>;
using Mat21f = MatNf<2, 1>;
using Mat22f = MatNf<2, 2>;
using Mat23f = MatNf<2, 3>;
using Mat31f = MatNf<3, 1>;
using Mat32f = MatNf<3, 2>;
using Mat33f = MatNf<3, 3>;

using Mat1f = Mat11f;
using Mat2f = Mat22f;
using Mat3f = Mat33f;

template <typename T, size_t Rows, size_t Cols>
struct Mat {
  static constexpr size_t MaxStackSize = 4;
  static constexpr size_t Dynamic = 0;
  static constexpr bool UseStack = (Rows <= MaxStackSize && Rows != Dynamic);
  using StorageType =
      typename std::conditional<UseStack, std::array<Vec<T, Cols>, Rows>,
                                std::vector<Vec<T, Cols>>>::type;

  StorageType data;

  constexpr Mat() = default;

  constexpr Mat(const Mat& other) : data(other.data) {}

  constexpr Mat(Mat&& other) noexcept : data(other.data) {}

  constexpr Mat& operator=(const Mat& other) {
    if constexpr (!UseStack) {
      data.resize(other.rows());
    }
    std::copy(std::begin(other.data), std::end(other.data), std::begin(data));
    return *this;
  }

  constexpr explicit Mat(T value) { data.fill(Vec<T, Cols>(value)); }

  Mat(size_t rows, size_t cols, T val = 0) {
    static_assert(!UseStack, "Cannot resize stack allocated matrix.");
    data.resize(rows, Vecf(cols, val));
  }

  constexpr Mat(std::initializer_list<std::initializer_list<T>> values) {
    if (values.size() != Rows) {
      throw std::runtime_error("Invalid initializer list length.");
    }

    size_t i = 0;
    for (const auto& val : values) {
      if constexpr (UseStack) {
        data[i++] = Vec<T, Cols>(val);
      } else {
        data.push_back(Vec<T, Cols>(val));
      }
    }
  }

  static constexpr Mat one() { return Mat<T, Rows, Cols>(1); }

  static constexpr Mat zero() { return Mat<T, Rows, Cols>(0); }

  static constexpr Mat<T, Rows, Cols> identity() {
    static_assert(Rows == Cols, "Identity matrix must be square.");
    Mat<T, Rows, Cols> identity(0);
    for (size_t i = 0; i < Rows; ++i) {
      identity[i][i] = 1;
    }
    return identity;
  }

  constexpr Vec<size_t, 2> size() const { return {rows(), cols()}; }

  [[nodiscard]] constexpr size_t rows() const {
    if constexpr (UseStack) {
      return Rows;
    } else {
      return data.size();
    }
  }

  [[nodiscard]] constexpr size_t cols() const {
    if constexpr (Cols != Dynamic) {
      return Cols;
    } else if (data.size() != 0) {
      return data[0].size();
    } else {
      return 0;
    }
  }

  void resize(size_t rows, size_t cols) {
    if constexpr (UseStack) {
      throw std::runtime_error("Cannot resize stack allocated matrix.");
    } else {
      data.resize(rows);
      for (auto& vec : data) {
        vec.resize(cols);
      }
    }
  }

  Vec<T, Cols>& operator[](size_t index) { return data[index]; }

  const Vec<T, Cols>& operator[](size_t index) const { return data[index]; }

  Vec<T, Rows> col(size_t index) const {
    Vec<T, Rows> result;
    if constexpr (!UseStack) {
      result.resize(rows());
    }
    for (int i = 0; i < rows(); i++) {
      result[i] = data[i][index];
    }
    return result;
  }

  Mat transpose() const {
    Mat<T, Cols, Rows> result;
    if constexpr (!UseStack) {
      result.resize(cols(), rows());
    }
    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = 0; j < cols(); ++j) {
        result[j][i] = data[i][j];
      }
    }
    return result;
  }

  Mat& transposed() {
    for (size_t i = 0; i < rows(); ++i) {
      for (size_t j = i + 1; j < cols(); ++j) {
        std::swap(data[i][j], data[j][i]);
      }
    }
    return *this;
  }

  template <size_t ColsB>
  Mat multiply(const Mat<T, Cols, ColsB>& other) const {
    Mat<T, Rows, ColsB> result;
    if constexpr (!UseStack) {
      result.resize(rows(), other.cols());
    }

    size_t i, j, k;
#ifdef __OMP_H
#pragma omp parallel for private(i, j, k) shared(result, data, other)
#endif
    for (i = 0; i < rows(); ++i) {
      for (k = 0; k < other.cols(); ++k) {
        T sum = 0;
        for (j = 0; j < cols(); ++j) {
          sum += data[i][j] * other[j][k];
        }
        result[i][k] = sum;
      }
    }
    return result;
  }

  Vec<T, Rows> multiply(const Vec<T, Cols>& vec) const {
    Vec<T, Rows> result;
    if constexpr (!UseStack) {
      if (cols() != vec.size()) {
        throw std::runtime_error("Invalid vector size.");
      }
      result.resize(rows());
    }
    for (size_t i = 0; i < rows(); ++i) {
      T sum = T();
      for (size_t j = 0; j < cols(); ++j) {
        sum += data[i][j] * vec[j];
      }
      result[i] = sum;
    }
    return result;
  }

  // Map each row vector of a matrix to a new matrix using func
  template <typename U = T>
  Mat map(std::function<Vec<U, Cols>(const Vec<T, Cols>&)> func) const {
    Mat<U, Rows, Cols> result;
    if constexpr (!UseStack) {
      result.resize(rows(), cols());
    }
    for (size_t i = 0; i < rows(); i++) {
      result[i] = func(data[i]);
    }
    return result;
  }

  // Map each row vector of a matrix to a new matrix using func
  template <typename U = T>
  Mat map(std::function<Vec<U, Cols>(const Vec<T, Cols>&, size_t)> func) const {
    Mat<U, Rows, Cols> result;
    if constexpr (!UseStack) {
      result.resize(rows(), cols());
    }
    for (size_t i = 0; i < rows(); i++) {
      result[i] = func(data[i], i);
    }
    return result;
  }

  // Apply func to each row vector of a matrix
  Mat& apply(std::function<void(Vec<T, Cols>&)> func) {
    for (Vec<T, Cols>& value : data) {
      func(value);
    }
    return *this;
  }

  // Apply func to each row vector of a matrix
  Mat& apply(std::function<void(Vec<T, Cols>&, size_t)> func) {
    for (size_t i = 0; i < rows(); i++) {
      func(data[i], i);
    }
    return *this;
  }

#define _mat_unary_op(op)                                          \
  Mat operator op() const {                                        \
    return map<T>([](const Vec<T, Cols>& vec) { return op vec; }); \
  }

  _mat_unary_op(+);
  _mat_unary_op(-);
  _mat_unary_op(~);
  _mat_unary_op(!);
#undef _mat_unary_op

#define _mat_op_scalar(op)                                                   \
  friend Mat operator op(const T& scalar, const Mat<T, Rows, Cols>& mat) {   \
    return mat op scalar;                                                    \
  }                                                                          \
  Mat operator op(const T& scalar) const {                                   \
    return map<T>(                                                           \
        [&](const Vec<T, Cols>& vec, size_t i) { return vec op scalar; });   \
  }                                                                          \
  Mat& operator op##=(const T& scalar) {                                     \
    return apply([&](Vec<T, Cols>& vec, size_t i) { vec = vec op scalar; }); \
  }

  _mat_op_scalar(+);
  _mat_op_scalar(-);
  _mat_op_scalar(*);
  _mat_op_scalar(/);
  _mat_op_scalar(%);
#undef _mat_op_scalar

#define _mat_op_mat(op)                                                        \
  Mat operator op(const Mat<T, Rows, Cols>& other) const {                     \
    return map<T>(                                                             \
        [&](const Vec<T, Cols>& vec, size_t i) { return vec op other[i]; });   \
  }                                                                            \
  Mat& operator op##=(const Mat<T, Rows, Cols>& other) {                       \
    return apply([&](Vec<T, Cols>& vec, size_t i) { vec = vec op other[i]; }); \
  }

  _mat_op_mat(+);
  _mat_op_mat(-);
  _mat_op_mat(*);
  _mat_op_mat(/);
  _mat_op_mat(%);
#undef _mat_op_mat

  bool operator==(const Mat<T, Rows, Cols>& other) const {
    for (size_t i = 0; i < rows(); i++) {
      if (this->data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Mat<T, Rows, Cols>& other) const {
    return !(*this == other);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const Mat<T, Rows, Cols>& mat) {
    os << "[";
    for (std::size_t i = 0; i < mat.rows() - 1; i++) {
      os << mat[i] << ",\n";
    }
    os << mat[mat.rows() - 1] << "]";
    return os;
  }

  friend std::istream& operator>>(std::istream& is, Mat<T, Rows, Cols>& mat) {
    is.ignore(1);
    for (size_t i = 0; i < mat.rows(); ++i) {
      is >> mat[i];
      if (i != mat.rows() - 1) {
        is.ignore(2);
      }
    }
    if (is.fail()) {
      throw std::runtime_error("Failed to read Vec");
    }
    return is;
  }

  static Mat<T, Rows, Cols> random(double min, double max) {
    Mat<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows; ++i) {
      result[i] = Vec<T, Cols>::random(min, max);
    }
    return result;
  }

  static Mat<T, Rows, Cols> random() {
    Mat<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows; ++i) {
      result[i] = Vec<T, Cols>::random();
    }
    return result;
  }
};

template <typename T, std::size_t Dim0, std::size_t... Dims,
          std::enable_if_t<((Dim0 == Dims) && ...), bool> = true>
Mat(T const (&)[Dim0], T const (&... arrs)[Dims])
    -> Mat<T, 1u + sizeof...(Dims), Dim0>;
