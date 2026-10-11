#include <tmol/utility/tensor/native_library.hh>
#include <tmol/score/common/whole_pose_scoring.hh>

namespace tmol::native {
namespace {
struct Exchange {
  std::vector<stable::Tensor> inputs, outputs;
};
thread_local Exchange* active_exchange = nullptr;
std::vector<stable::Tensor> exchange_tensors(
    std::vector<stable::Tensor> inputs) {
  STD_TORCH_CHECK(
      active_exchange, "Tensor exchange requires an active native call");
  active_exchange->inputs = std::move(inputs);
  return std::move(active_exchange->outputs);
}
py::object exchange_python(Exchange& exchange, py::handle inputs) {
  struct Scope {
    Exchange* previous;
    explicit Scope(Exchange& value) : previous(active_exchange) {
      active_exchange = &value;
    }
    ~Scope() { active_exchange = previous; }
  } scope(exchange);
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  auto& op =
      storage
          .call_once_and_store_result([] {
            using Library = stable::detail::StableLibrary;
            static Library definitions(
                Library::Kind::DEF,
                TMOL_INTEROP_NAMESPACE,
                nullptr,
                __FILE__,
                __LINE__);
            static Library implementations(
                Library::Kind::IMPL,
                TMOL_INTEROP_NAMESPACE,
                "CompositeImplicitAutograd",
                __FILE__,
                __LINE__);
            definitions.def("exchange(Tensor[] inputs) -> Tensor[]");
            implementations.impl("exchange", TORCH_BOX(&exchange_tensors));
            return py::module_::import("torch")
                .attr("ops")
                .attr(TMOL_INTEROP_NAMESPACE)
                .attr("exchange");
          })
          .get_stored();
  return call_one(op, inputs);
}
}  // namespace
stable::Tensor import_tensor_python(py::handle obj) {
  Exchange exchange;
  exchange_python(exchange, py::make_tuple(obj));
  return std::move(exchange.inputs.at(0));
}
py::object export_tensor_python(const stable::Tensor& tensor) {
  Exchange exchange;
  exchange.outputs.push_back(tensor);
  return exchange_python(exchange, py::tuple())[py::int_(0)];
}

// Batch the older-runtime fallback so saved tensors cross the dispatcher once.
py::tuple to_python_tensors(const std::vector<Tensor>& tensors) {
  py::tuple result(tensors.size());
  Exchange exchange;
  std::vector<size_t> pending;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    if (!tensor.defined() || (tensor.original && tensor.original->object)
        || tensor_api().export_tensor) {
      result[i] = to_python(tensor);
    } else {
      exchange.outputs.push_back(tensor.handle());
      pending.push_back(i);
    }
  }
  if (!pending.empty()) {
    auto objects = exchange_python(exchange, py::tuple());
    for (size_t j = 0; j < pending.size(); ++j) {
      size_t i = pending[j];
      result[i] = objects[py::int_(j)];
    }
  }
  return result;
}
std::vector<Tensor> from_python_tensors(py::handle objects) {
  auto sequence = py::reinterpret_borrow<py::sequence>(objects);
  std::vector<Tensor> result(sequence.size());
  const auto& api = tensor_api();
  if (api.import_tensor) {
    for (size_t i = 0; i < result.size(); ++i)
      result[i] = from_python(sequence[i]);
    return result;
  }
  py::list inputs;
  std::vector<size_t> pending;
  for (size_t i = 0; i < result.size(); ++i) {
    auto object = sequence[i];
    if (object.is_none()) continue;
    inputs.append(
        py::cast<bool>(call_one(api.is_neg, object))
            ? call_one(api.resolve_neg, object)
            : py::reinterpret_borrow<py::object>(object));
    pending.push_back(i);
  }
  if (!pending.empty()) {
    Exchange exchange;
    exchange_python(exchange, inputs);
    for (size_t j = 0; j < pending.size(); ++j) {
      size_t i = pending[j];
      auto& tensor = result[i];
      tensor = Tensor(std::move(exchange.inputs[j]));
      if (py::cast<bool>(call_one(api.is_neg, sequence[i]))) {
        tensor.original = std::make_shared<PythonOwner>();
        own_python(*tensor.original, sequence[i]);
      }
      tensor.needs_grad =
          GradMode::is_enabled()
          && py::cast<bool>(sequence[i].attr(api.requires_grad));
    }
  }
  return result;
}
Tensor Tensor::reshape(IntList shape) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("reshape")(
        py::cast(std::vector<int64_t>(shape.begin(), shape.end()))));
  }
  return stable::reshape(handle(), shape);
}
Tensor Tensor::narrow(int64_t d, int64_t start, int64_t length) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("narrow")(d, start, length));
  }
  auto input = handle();
  return stable::narrow(input, d, start, length);
}
Tensor Tensor::select(int64_t d, int64_t i) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("select")(d, i));
  }
  return stable::select(handle(), d, i);
}
Tensor Tensor::contiguous() const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("contiguous")());
  }
  return stable::contiguous(handle());
}
Tensor Tensor::unsqueeze(int64_t d) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("unsqueeze")(d));
  }
  return stable::unsqueeze(handle(), d);
}
Tensor Tensor::squeeze(int64_t d) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("squeeze")(d));
  }
  return stable::squeeze(handle(), d);
}
Tensor Tensor::sum(IntList dims) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("sum")(
        py::cast(std::vector<int64_t>(dims.begin(), dims.end()))));
  }
  return stable::sum(handle(), dims);
}
Tensor Tensor::index_select(int64_t d, const Tensor& index) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(
        to_python(*this).attr("index_select")(d, to_python(index)));
  }
  return stable::index_select(handle(), d, index.handle());
}
Tensor operator*(const Tensor& a, const Tensor& b) {
  if ((a.needs_grad || b.needs_grad) && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(a).attr("__mul__")(to_python(b)));
  }
  std::array<StableIValue, 2> stack{
      stable::detail::from(a.handle()), stable::detail::from(b.handle())};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return stable::detail::to<stable::Tensor>(stack[0]);
}
}  // namespace tmol::native
#ifdef TMOL_MAIN_MODULE
PYBIND11_MODULE(_C, m) {
  tmol::native::bind_libraries(m);
  using tmol::native::autograd::ScoreCapture;
  py::class_<ScoreCapture>(m, "_ScoreCapture")
      .def(py::init<>())
      .def(
          "select_term",
          [](ScoreCapture& self, int64_t offset, bool enabled) {
            self.offset = offset;
            self.enabled = enabled;
          })
      .def_readonly("offsets", &ScoreCapture::offsets)
      .def_property_readonly(
          "derivatives",
          [](ScoreCapture& self) {
            return tmol::native::to_python_tensors(self.derivatives);
          })
      .def(
          "__enter__",
          [](ScoreCapture& self) -> ScoreCapture& {
            self.previous = tmol::native::autograd::score_capture;
            tmol::native::autograd::score_capture = &self;
            return self;
          },
          py::return_value_policy::reference_internal)
      .def("__exit__", [](ScoreCapture& self, py::args) {
        tmol::native::autograd::score_capture = self.previous;
      });
#ifdef WITH_CUDA
  // The dispatcher imports the complete tensor list directly, including on
  // older runtimes without the Python tensor C API.
  static tmol::native::Library gradients(TMOL_INTEROP_NAMESPACE);
  gradients.def(
      "score_gradient",
      TMOL_FN(&tmol::score::common::accumulate_score_gradients_cuda));
  m.attr("_score_gradient") = py::module_::import("torch")
                                  .attr("ops")
                                  .attr(TMOL_INTEROP_NAMESPACE)
                                  .attr("score_gradient");
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}

#endif
