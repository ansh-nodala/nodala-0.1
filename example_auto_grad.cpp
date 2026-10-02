#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

template <typename T, typename T2> struct TensorNode {
  T data;
  T2 grad;
  std::function<void()> _backward;
  std::vector<std::shared_ptr<TensorNode<T, T2>>> _prev;
  int in_degree = 0;

  TensorNode(T d) : data(d), grad(T2{}) {
    _backward = []() {};
  }
};

template <typename T, typename T2> class Tensor {
private:
  std::shared_ptr<TensorNode<T, T2>> node;

  Tensor(std::shared_ptr<TensorNode<T, T2>> n) : node(n) {};

public:
  Tensor(T val) { node = std::make_shared<TensorNode<T, T2>>(val); }

  T data() const { return node->data; }
  T2 grad() const { return node->grad; }

  Tensor operator+(const Tensor &other) const {
    auto out_node =
        std::make_shared<TensorNode<T, T2>>(other.data() + this->data());
    out_node->_prev = {this->node, other.node};

    auto self_node = this->node;
    auto other_node = other.node;
    auto out_ptr = out_node.get();

    out_node->_backward = [self_node, other_node, out_ptr]() {
      self_node->grad += out_ptr->grad;
      other_node->grad += out_ptr->grad;
    };

    return Tensor(out_node);
  }

  Tensor operator*(const Tensor &other) const {
    auto out_node =
        std::make_shared<TensorNode<T, T2>>(other.data() * this->data());
    out_node->_prev = {this->node, other.node};

    auto self_node = this->node;
    auto other_node = other.node;

    auto out_ptr = out_node.get();

    out_node->_backward = [self_node, other_node, out_ptr]() {
      self_node->grad += other_node->data * out_ptr->grad;
      other_node->grad += self_node->data * out_ptr->grad;
    };

    return Tensor(out_node);
  }

  Tensor operator-(const Tensor &other) const {
    auto out_node =
        std::make_shared<TensorNode<T, T2>>(this->data() - other.data());
    auto self_node = this->node;
    auto other_node = other.node;
    auto out_ptr = out_node.get();

    out_node->_prev = {self_node, other_node};
    out_node->_backward = [self_node, other_node, out_ptr]() {
      self_node->grad += out_ptr->grad;
      other_node->grad -= out_ptr->grad;
    };

    return Tensor(out_node);
  }

  Tensor relu() const {
    T val = this->data() > 0 ? this->data() : T{0};
    auto out_node = std::make_shared<TensorNode<T, T2>>(val);
    out_node->_prev = {this->node};

    auto self_node = this->node;
    auto out_ptr = out_node.get();

    out_node->_backward = [self_node, out_ptr]() {
      if (self_node->data > 0) {
        self_node->grad += out_ptr->grad;
      }
    };

    return Tensor(out_node);
  }

  void backward(T2 initial_grad = T2{1}) {
    std::unordered_set<std::shared_ptr<TensorNode<T, T2>>> visited;
    std::queue<std::shared_ptr<TensorNode<T, T2>>> q;

    std::function<void(std::shared_ptr<TensorNode<T, T2>>)> build_topo =
        [&](std::shared_ptr<TensorNode<T, T2>> v) {
          if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto child : v->_prev) {
              child->in_degree++;
              build_topo(child);
            }
          }
        };

    build_topo(node);

    node->grad = initial_grad;

    q.push(node);

    while (!q.empty()) {
      auto current = q.front();
      q.pop();
      current->_backward();
      for (auto child : current->_prev) {
        child->in_degree--;
        if (child->in_degree == 0) {
          q.push(child);
        }
      }
    }
  }
};

PYBIND11_MODULE(micrograd_cpp, m) {
  m.doc() = "a c++ micrograd engine for calcluting the gradient of function";
  py::class_<Tensor<float, float>>(m, "Tensor")
      .def(py::init<float>())
      .def("data", &Tensor<float, float>::data)
      .def("grad", &Tensor<float, float>::grad)
      .def("ReLU", &Tensor<float, float>::relu)
      .def("backward", &Tensor<float, float>::backward,
           py::arg("initial_grad") = 1.0f)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def("__radd__",
           [](const Tensor<float, float> &self,
              const Tensor<float, float> &other) { return self + other; })
      .def("__rsub__",
           [](const Tensor<float, float> &self,
              const Tensor<float, float> &other) { return other - self; })
      .def("__rmul__",
           [](const Tensor<float, float> &self,
              const Tensor<float, float> &other) { return self * other; })
      .def("__repr__", [](const Tensor<float, float> &t) {
        return "Tensor([" + std::to_string(t.data()) + "])";
      });

  py::implicitly_convertible<float, Tensor<float, float>>();
  py::implicitly_convertible<int, Tensor<float, float>>();
}
/***
int main() {

  Tensor<float, float> a(1.2);
  Tensor<float, float> b(2.1);
  Tensor<float, float> c(1.0);

  auto res = a + b * c;

  auto loss = res.data() - 1.15;
  auto relu_res = res.relu();

  std::cout << "ReLU res: " << relu_res.grad() << "data" << relu_res.data()
            << std::endl;

  res.backward(loss);

  std::cout << res.grad() << std::endl;
  std::cout << res.data() << std::endl;
  return 0;
}

***/
