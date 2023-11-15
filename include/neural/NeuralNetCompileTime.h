#pragma once
#include "Common.h"
#include "Mat.h"
#include "Vec.h"
template <typename T, size_t InSize, size_t OutSize>
struct Layer {
  Mat<T, OutSize, InSize> weights;
  Vec<T, OutSize> biases;
  bool use_relu;

  Layer() = default;
  Layer& operator=(const Layer& other) {
    weights = other.weights;
    biases = other.biases;
    use_relu = other.use_relu;
    return *this;
  }

  Layer(const Mat<T, OutSize, InSize>& w, const Vec<T, OutSize>& b, bool relu)
      : weights(w), biases(b), use_relu(relu) {}

  Vec<T, OutSize> feedforward(const Vec<T, InSize>& input) const {
    Vec<T, OutSize> output = weights.multiply(input) + biases;
    if (use_relu) {
      output.apply([](T& val) { val = Math::relu(val); });
    }
    return output;
  }
};

template <typename T, size_t... Sizes>
class NeuralNet {
 private:
  static_assert((sizeof...(Sizes) >= 2) && (sizeof...(Sizes) % 2 == 0),
                "NeuralNet must have at least two sizes (input and output).");

  template <size_t InSize, size_t OutSize, size_t... Rest>
  struct LayerTuple {
    using type =
        decltype(std::tuple_cat(std::tuple<Layer<T, InSize, OutSize>>{},
                                typename LayerTuple<Rest...>::type{}));
  };

  template <size_t InSize, size_t OutSize>
  struct LayerTuple<InSize, OutSize> {
    using type = std::tuple<Layer<T, InSize, OutSize>>;
  };

  typename LayerTuple<Sizes...>::type layers;

 public:
  template <size_t Index, typename W, typename B>
  void setLayer(const W& weights, const B& biases, bool relu = true) {
    static_assert(W::rows() == B::size(),
                  "Weights and biases must be compatible.");
    std::get<Index>(layers) =
        Layer<T, W::cols(), W::rows()>(weights, biases, relu);
  }

  template <size_t LayerIndex = 0, typename InputVec>
  auto evaluate(const InputVec& input) {
    if constexpr (LayerIndex < std::tuple_size_v<decltype(layers)> - 1) {
      auto& layer = std::get<LayerIndex>(layers);
      auto output = layer.feedforward(input);
      return evaluate<LayerIndex + 1>(output);
    } else {
      return std::get<LayerIndex>(layers).feedforward(input);
    }
  }

  template <size_t Index>
  auto& getLayer() {
    return std::get<Index>(layers);
  }
};
