## Learn Pytorch Internals -- Autograd



### Variable

### Engine
When we use
```
loss.backward()
```
it calls
```
def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    //check ...
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
```

```
def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensors] = None) -> None:

    Variable._execution_engine.run_backward(
        tensors, grad_tensors_, retain_graph, create_graph, inputs,
        allow_unreachable=True, accumulate_grad=True)  

```
Pytorch use CPython or Pybind to extent python in C++, for more detail, please refer to reference. Here we can see what function is called in the methods table, it calls the THPEngine_run_backward, which is a C++ function.
```
static struct PyMethodDef THPEngine_methods[] = {
  {(char*)"run_backward",
    castPyCFunctionWithKeywords(THPEngine_run_backward),
    METH_VARARGS | METH_KEYWORDS, nullptr},
  {(char*)"queue_callback", THPEngine_queue_callback, METH_O, nullptr},
  {(char*)"is_checkpoint_valid", THPEngine_is_checkpoint_valid, METH_NOARGS, nullptr},
  {nullptr}
};
```
Now in Now in THPEngine_run_backward, we can find out what `backward()` exactly do.
```
PyObject *THPEngine_run_backward(PyObject *self, PyObject *args, PyObject *kwargs)
{
if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OObb|Obb", (char**)accepted_kwargs,
        &tensors, &grad_tensors, &keep_graph, &create_graph, &inputs, &allow_unreachable, &accumulate_grad))
    return nullptr;
//edge_list: vector
edge_list roots;
roots.reserve(num_tensors);
variable_list grads;
grads.reserve(num_tensors);
for(const auto i : c10::irange(num_tensors)) {

  // roots
  auto gradient_edge = torch::autograd::impl::gradient_edge(variable);
  roots.push_back(std::move(gradient_edge));

  // grads
  PyObject *grad = PyTuple_GET_ITEM(grad_tensors, i);
  const Variable& grad_var = THPVariable_Unpack(grad);
  grads.push_back(grad_var);


```
1. What is `edge_list`

    ```
    using edge_list = std::vector<Edge>;
    ```
    Edge is a struct:
    ```
    struct Edge {
    Edge() noexcept : function(nullptr), input_nr(0) {}

    Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
        : function(std::move(function_)), input_nr(input_nr_) {}
    ```
    In Pytorch graph, each edge stores the function (`function`) that the edge points at, and it also stores index(`input_nr`) of this function, since function could have more than one input, i.e. `input_nr` indicates that this edge is which input to the function.

2. gradient_edge

    If grad_fn is null (as a leaf node),  the gradient function is a gradient accumulator, which will accumulate its inputs into the grad property of the variable. Note that only variables which have `requires_grad = True` can have gradient accumulators.

    ```
    Edge gradient_edge(const Variable& self) {
        if (const auto& gradient = self.grad_fn()) {
        return Edge(gradient, self.output_nr());
        } else {
        return Edge(grad_accumulator(self), 0);
        }
    }
    ```
3. grads

     `grads = None` by default, however if you call loss.backward(gradient = torch.rand([x,x...x])), it would saves gradient tensor 


```
???
  std::vector<Edge> output_edges;
  if (inputs != nullptr) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    output_edges.reserve(num_inputs);

    // Here we will not step into this loop, since num_inputs=0
    for (const auto i : c10::irange(num_inputs)) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      const auto& tensor = THPVariable_Unpack(input);
      const auto output_nr = tensor.output_nr();
      auto grad_fn = tensor.grad_fn();
      if (!grad_fn) {
        grad_fn = torch::autograd::impl::try_get_grad_accumulator(tensor);
      }
      if (accumulate_grad) {
        tensor.retain_grad();
      }
      if (!grad_fn) {
        output_edges.emplace_back(std::make_shared<Identity>(), 0);
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }


```

```
  variable_list outputs;
  {
    pybind11::gil_scoped_release no_gil; 
    auto& engine = python::PythonEngine::get_python_engine();
    outputs = engine.execute(roots, grads, keep_graph, create_graph,accumulate_grad, output_edges);
  }
```


1. Rekease [GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc. html#global-interpreter-lock-gil)

    The autograd engine was called while holding the GIL, the autograd engine is an expensive operation that does not require the GIL to be held so you should release it with `pybind11::gil_scoped_release no_gil;`
  
2. Engine.execute()

    ```
    variable_list PythonEngine::execute(const edge_list& roots, const variable_list& inputs,
      bool keep_graph, bool create_graph, bool accumulate_grad, const edge_list& outputs) {
        return Engine::execute(roots, inputs, keep_graph, create_graph, accumulate_grad, outputs);
      }
    ```

### Engine::execute

```
auto Engine::execute(const edge_list& roots,
                  const variable_list& inputs,
                  bool keep_graph,
                  bool create_graph,
                  bool accumulate_grad,
                  const edge_list& outputs) -> variable_list {

  //---------------------------1-------------------------------
  validate_outputs(roots, const_cast<variable_list&>(inputs), [](const std::string& msg) {
    return msg;
  });

  // A fresh first time Engine::execute call should start on the CPU device, initialize
  init_local_ready_queue();
  bool not_reentrant_backward_call = worker_device == NO_DEVICE;??
  //---------------------------2-------------------------------
              //------------tip 1-----------
  auto graph_task = std::make_shared<GraphTask>(
      /* keep_graph */ keep_graph,
      /* create_graph */ create_graph,
      /* depth */ not_reentrant_backward_call ? 0 : total_depth + 1,
      /* cpu_ready_queue */ local_ready_queue);

  // If we receive a single root, skip creating extra root node
  bool skip_dummy_node = roots.size() == 1;
  auto graph_root = skip_dummy_node ?
    roots.at(0).function :
    std::make_shared<GraphRoot>(roots, inputs);

  auto min_topo_nr = compute_min_topological_nr(outputs);
  // Now compute the dependencies for all executable functions
  compute_dependencies(graph_root.get(), *graph_task, min_topo_nr);

  if (!outputs.empty()) {
    graph_task->init_to_execute(*graph_root, outputs, accumulate_grad, min_topo_nr);
  }

  // Queue the root
  if (skip_dummy_node) {
    InputBuffer input_buffer(roots.at(0).function->num_inputs());
    auto input = inputs.at(0);

    const auto input_stream = InputMetadata(input).stream();
    const auto opt_next_stream = roots.at(0).function->stream(c10::DeviceType::CUDA);
    input_buffer.add(roots.at(0).input_nr,
                      std::move(input),
                      input_stream,
                      opt_next_stream);

    execute_with_graph_task(graph_task, graph_root, std::move(input_buffer));
  } else {
    execute_with_graph_task(graph_task, graph_root, InputBuffer(variable_list()));
  }
  // Avoid a refcount bump for the Future, since we check for refcount in
  // DistEngine (see TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1)
  // in dist_engine.cpp).
  auto& fut = graph_task->future_result_;
  fut->wait();
  return fut->value().toTensorVector();
}
```


1.Local ready queue

  A fresh first time Engine::execute call should start on the CPU device, initialize a new thread local ready queue on CPU or reuse the existing one (if there is one allocated already, i.e. consecutive backward calls)


    1.`init_local_ready_queue()`
      ```
      void Engine::init_local_ready_queue(std::shared_ptr<ReadyQueue> ready_queue) {
        if (ready_queue) {
          // if ready_queue provided in the caller, use the caller's ready_queue to initialize local_ready_queue
          local_ready_queue = std::move(ready_queue);
        } else if (!local_ready_queue){
          // otherwise if local_ready_queue not allocated, allocate a new ready_queue
          local_ready_queue = std::make_shared<ReadyQueue>();
        }
      }
      ```

    2.ReadyQueue

      ReadyQueue uses priority queue to maintain NodeTasks.
      ```
      struct ReadyQueue {
        private:
          ...
          std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime> heap_;
          ...};
      ```

2.GraphTask 

  GraphTask holds metadata needed for execution of backward(), e.g: local_ready_queue.
  ```
  struct GraphTask: std::enable_shared_from_this<GraphTask> {
    std::shared_ptr<ReadyQueue> cpu_ready_queue_;
  ```
  CPU threads are dedicated to processing CPU work for the backward they invoked. So any given graph task maintains its own cpu_ready_queue_ where you should send work for it to be done. We memoize the cpu_ready_queue_ per GraphTask so that we know which ready queue we should push to if we are on device thread (i.e. GPU) and but next NodeTask should be run on CPU.






    tip:
    1.make_shared
      [make_shared V.S. shared_ptr](https://www.jianshu.com/p/03eea8262c11)


  To understand the [reentrant](https://www.cnblogs.com/clover-toeic/p/3738464.html) backwards problem, we have to notice two aspects of how the autograd engine is implemented today:

    1. When you call Engine::execute(), you want to block until differentiation finishes so that you can get the final result variables of the backwards pass.
    2. The engine operates by having a single worker thread per work queue, and every work queue is pinned to a specific device where the operation is executed.







[reference](https://zhuanlan.zhihu.com/p/336599887)




```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/lalalazy12/LearnPytorch/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
