#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <mutex>
#include <thread>

#include "Common.h"
#include "Mat.h"
#include "Vec.h"

struct Layer {
    Matf weights;
    Vecf biases;
    bool relu;

    Layer() = default;

    Vecf forward(const Vecf& input) const {
        Vecf result = weights.multiply(input) + biases;
        if (relu) {
            result.apply([](float& x) { x = x > 0 ? x : 0; });
        }
        return result;
    }

    Vecf backward(const Vecf& output, const Vecf& outputGradient) {
        Vecf inputGradient(weights.cols(), 0.0);

        for (size_t i = 0; i < weights.rows(); ++i) {
            double activationGradient = (relu && output[i]) <= 0 ? 0 : 1;
            for (size_t j = 0; j < weights.cols(); ++j) {
                inputGradient[j] += weights[i][j] * outputGradient[i];
            }
        }

        return inputGradient;
    }

    Vecf computeGradients(const Vecf& input) const {
        Vecf output = forward(input);
        Vecf grad(input.size(), 0.0);
        for (int i = 0; i < output.size(); ++i) {
            grad += ((output[i] > 0) - (output[i] < 0)) * weights[i];
        }
        return grad;
    }

    friend std::istream& operator>>(std::istream& is, Layer& l) {
        is.ignore(9);  // "Rows 1: "
        int rows;
        is >> rows;
        is.ignore(9);  // "Cols 1: "
        int cols;
        is >> cols;

        l.weights.resize(rows, cols);
        l.biases.resize(rows);

        is.ignore(12);  // "Weights 1: "
        is >> l.weights;

        is.ignore(12);  // "Biases 1: "
        Matf biases(rows, 1);
        is >> biases;
        l.biases = biases.col(0);
        is.ignore(10);  // "Relu 1: "
        string relu;
        is >> relu;
        l.relu = (relu == "true");
        return is;
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& l) {
        cout << "Rows: " << l.weights.rows() << endl;
        cout << "Cols: " << l.weights.cols() << endl;
        cout << "Weights: " << l.weights << endl;
        cout << "Biases: " << l.biases << endl;
        return os;
    }
};

struct NeuralNet {
    vector<Layer> layers;
    float learningRate;
    function<float(const Vecf&)> cost_func;
    function<Vecf(const Vecf&)> cost_func_d;

    NeuralNet() { learningRate = 10; }

    void addLayer(const Layer& layer) { layers.push_back(layer); }

    Vecf evaluate(const Vecf& input) const {
        Vecf output = input;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    Vecf random_search(size_t iterations, float epsilon) {
        Vecf bestInput(layers[0].weights.cols(), 0);
        float bestLoss = std::numeric_limits<float>::max();
        for (int i = 0; i < iterations; ++i) {
            Vecf input(layers[0].weights.cols(), 0);
            for (int j = 0; j < input.size(); ++j) {
                input[j] = Math::random_float(-10000.0f, 10000.0f);
            }
            Vecf output = evaluate(input);
            float loss = output.abs_sum();
            if (loss < bestLoss) {
                bestLoss = loss;
                bestInput = input;
            }
            if (loss < epsilon) {
                break;
            }
        }
        return bestInput;
    }

    Vecf optimize(const Vecf& initInput, size_t iterations,
                  float epsilon) const {
        Vecf input = initInput;
        float a = learningRate;
        float lastLoss = std::numeric_limits<float>::max();
        for (int i = 0; i < iterations; ++i) {
            Vecf output = evaluate(input);
            Vecf grad = output.map<float>([](float val) {
                return val > 0 ? 1.0f : (val < 0 ? -1.0f : 0.0f);
            });
            float loss = cost_func(output);
            if (loss < epsilon) {
                break;
            }
            if (!input.fold<bool>(
                    [](bool b, float v) {
                        return (std::isnan(v) || std::isinf(v)) && b;
                    },
                    true)) {
                break;
            }
            if (loss > lastLoss) {
                a /= 2;
            }
            lastLoss = loss;

            input -= a * grad;
        }
        return input;
    }

    Vecf optimize(size_t samples, size_t iterations, float epsilon) {
        Vecf bestInput(layers[0].weights.cols(), 0);
        float bestLoss = std::numeric_limits<float>::max();

        for (size_t i = 0; i < iterations; i++) {
            Vecf input(layers[0].weights.cols(), 0);
            for (int j = 0; j < input.size(); ++j) {
                input[j] = Math::random_float(-10000.0f, 10000.0f);
            }

            input = optimize(input, iterations, epsilon);
            float loss = cost_func(evaluate(input));

            if (!input.fold<bool>(
                    [](bool b, float v) {
                        return (std::isnan(v) || std::isinf(v)) && b;
                    },
                    true)) {
                loss = std::numeric_limits<float>::max();
            }
            if (loss < bestLoss) {
                bestInput = input;
                bestLoss = loss;
            }
            if (loss <= epsilon) {
                break;
            }
        }

        return bestInput;
    }
    
    Vecf parallel_optimize_with_ramdom_seed(size_t samples, size_t iterations,
                                            float epsilon) {
        Vecf bestInput(layers[0].weights.cols(), 0);
        float bestLoss = std::numeric_limits<float>::max();
        std::mutex mtx;

        auto optimize_thread = [&](size_t thread_id) {
            Vecf localBestInput(layers[0].weights.cols(), 0);
            float localBestLoss = std::numeric_limits<float>::max();

            for (size_t i = 0; i < samples; i++) {
                Vecf input(layers[0].weights.cols(), 0);
                for (size_t j = 0; j < input.size(); ++j) {
                    input[j] = Math::random_float(-1000000.0f, 1000000.0f);
                }

                input = optimize(input, iterations, epsilon);
                float loss = cost_func(evaluate(input));
                if (loss < localBestLoss) {
                    localBestInput = input;
                    localBestLoss = loss;
                    if (loss <= epsilon) {
                        break;
                    }
                }
                if (i % 1000 == 0) {
                    if (bestLoss < epsilon) {
                        break;
                    }
                }
            }

            std::lock_guard<std::mutex> guard(mtx);
            if (localBestLoss < bestLoss) {
                bestInput = localBestInput;
                bestLoss = localBestLoss;
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
            threads.emplace_back(optimize_thread, i);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return bestInput;
    }

    friend std::istream& operator>>(std::istream& is, NeuralNet& nn) {
        is.ignore(9);  // "Layers:  "
        int layersCount;
        is >> layersCount;
        for (int i = 0; i < layersCount; ++i) {
            Layer layer;
            is >> layer;
            string line;
            nn.layers.push_back(layer);
        }
        string line;
        std::getline(is, line, '\n');
        std::getline(is, line, '\n');
        std::getline(is, line, '\n');
        std::getline(is, line, '\n');
        return is;
    }

    friend std::ostream& operator<<(std::ostream& os, NeuralNet& nn) {
        os << "======Network======" << endl;
        os << "Layers: " << nn.layers.size() << endl;
        for (size_t i = 0; i < nn.layers.size(); i++) {
            os << nn.layers[i];
        }
        return os;
    }
};
