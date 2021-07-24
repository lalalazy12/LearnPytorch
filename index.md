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


1. Release [GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc. html#global-interpreter-lock-gil)

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

  //---------------------------3-------------------------------
  // If we receive a single root, skip creating extra root node
  bool skip_dummy_node = roots.size() == 1;
  auto graph_root = skip_dummy_node ?
    roots.at(0).function :
    std::make_shared<GraphRoot>(roots, inputs);

  auto min_topo_nr = compute_min_topological_nr(outputs);
  // Now compute the dependencies for all executable functions
  compute_dependencies(graph_root.get(), *graph_task, min_topo_nr);

  //---------------------------4-------------------------------
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


1. Local ready queue

    A fresh first time Engine::execute call should start on the CPU device, initialize a new thread local ready queue on CPU or reuse the existing one (if there is one allocated already, i.e. consecutive backward calls)


    1. `init_local_ready_queue()`

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
    2. `ReadyQueue`

        ReadyQueue uses priority queue to maintain NodeTasks.
        ```
        struct ReadyQueue {
          private:
            ...
            std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime> heap_;
            ...};
        ```
    3. `NodeTask`

        ```
        struct NodeTask {
              std::weak_ptr<GraphTask> base,
              std::shared_ptr<Node> fn,
              InputBuffer inputs,
              bool isShutdownTask = false)
        };
        ```
        inputs buffer: Once all the dependencies are finished, we use the contents of this buffer to run the function.

2. GraphTask 

    GraphTask holds metadata needed for execution of backward(), e.g: local_ready_queue.
    ```
    struct GraphTask: std::enable_shared_from_this<GraphTask> {
      std::shared_ptr<ReadyQueue> cpu_ready_queue_;

      std::atomic<uint64_t> outstanding_tasks_{0};
      std::atomic_bool future_completed_{false};
      // It is safe to read grad_mode_ and keep_graph_ without synchronization
      bool keep_graph_;
      bool grad_mode_;

      // To protect reads/writes to not_ready_, dependencies_, captured_vars_,
      // has_error_, future_result_, cpu_ready_queue_, and leaf_streams.
      std::mutex mutex_;
      std::unordered_map<Node*, InputBuffer> not_ready_;
      std::unordered_map<Node*, int> dependencies_;
    ```
    CPU threads are dedicated to processing CPU work for the backward they invoked. So any given graph task maintains its own cpu_ready_queue_ where you should send work for it to be done. We memoize the cpu_ready_queue_ per GraphTask so that we know which ready queue we should push to if we are on device thread (i.e. GPU) and but next NodeTask should be run on CPU.


3. Compute_dependecies

    If the input is a single node, the single node is the starting point of the graph, otherwise a GraphRoot instance is created as the starting point of the graph.
    
    Calculate the number of dependencies of each node participating in the gradient calculation, implemented by BFS search for nodes in GraphTask through GraphRoot.
    
    ```
    auto Engine::compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr) -> void {
      std::unordered_set<Node*> seen;
      std::vector<Node*> queue { root };
      bool might_use_cuda = at::globalContext().hasCUDA();
      bool will_use_cuda = false;

      // Queue contains all nodes that will start propagating gradients.
      //----------BFS----------
      auto& dependencies = task.dependencies_;
      while (!queue.empty()) {
        auto fn = queue.back(); queue.pop_back();
        if (fn->topological_nr() < min_topo_nr) {
          continue;
        }
        for (const auto& edge : fn->next_edges()) {
          if (auto next_ptr = edge.function.get()) {
            dependencies[next_ptr] += 1;
            const bool was_inserted = seen.insert(next_ptr).second;
            if (was_inserted) queue.push_back(next_ptr);
    ```
    Dependencies is a member variable in GraphTask, the type: `std::unordered_map<Node*, int> dependencies_;`. After executing the above function, the number of keys in the dependencies is the same as the number of Nodes in the calculation graph, and the dependencies corresponding to each node can be regarded as the out-degree of the node in the forward calculation graph.
4. InputBuffer

    Inputs of grad function are stored in input_buffer.
    ```
    struct InputBuffer {
      explicit InputBuffer(size_t size)
        : buffer(size) {}
      InputBuffer(const InputBuffer& other) = delete;
      InputBuffer(InputBuffer&& other) = default;
      explicit InputBuffer(variable_list&& inputs): buffer(std::move(inputs)) {};

      void add(size_t pos,
              Variable&& var,
              const c10::optional<c10::Stream>& opt_producer_stream,
              const c10::optional<c10::Stream>& opt_consumer_stream);
    ```
    What is `add`: Accumulates the variable at a specified index. The optional CUDA streams determine which stream the accumulation is run on and how the addition is synchronized.

5. `execute_with_graph_task`

### Execute with Graph Task
```
c10::intrusive_ptr<at::ivalue::Future> Engine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {

  //---------------------------1-------------------------------
  initialize_device_threads_pool();
  // Lock mutex for GraphTask.
  std::unique_lock<std::mutex> lock(graph_task->mutex_);
  //---------------------------2-------------------------------
  auto queue = ready_queue(graph_task->cpu_ready_queue_, input_buffer.device());
  // worker_device == NO_DEVICE it's a CPU thread and it's trying to drive the
  // autograd engine with corresponding GraphTask, and its NOT a re-entrant call
  if (worker_device == NO_DEVICE) {
    // We set the worker_device to CPU_DEVICE only if worker_device was previously
    // NO_DEVICE. Setting it to CPU afterwards allow us to detect whether this is
    // a re-entrant call or not.
    set_device(CPU_DEVICE);

    // set the graph_task owner to the current device
    graph_task->owner_ = worker_device;

    // Now that all the non-thread safe fields of the graph_task have been populated,
    // we can enqueue it.
    queue->push(NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    // The owning thread start to drive the engine execution for any CPU task that
    // was just pushed or will be added later from other worker threads
    lock.unlock();
  //---------------------------3-------------------------------
    thread_main(graph_task);
    TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    // reset the worker_device after the completion of the graph_task, this is so
    // that the initial state of the engine remains the same across every backward()
    // or grad() call, we don't need to reset local_ready_queue as we could possibly
    // reuse it for new backward calls.
    worker_device = NO_DEVICE;
  } else {
    // If worker_device is any devices (i.e. CPU, CUDA): this is a re-entrant
    //    backward call from that device.
    graph_task->owner_ = worker_device;

    // Now that all the non-thread safe fields of the graph_task have been populated,
    // we can enqueue it.
    queue->push(NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    if (current_depth >= max_recursion_depth_) {
      // See Note [Reentrant backwards]
      // If reached the max depth, switch to a different thread
      add_thread_pool_task(graph_task);
    } else {
      // Total depth needs to be updated only in this codepath, since it is
      // not used in the block above (when we call add_thread_pool_task).
      // In the codepath above, GraphTask.reentrant_depth_ is used to
      // bootstrap total_depth in the other thread.
      ++total_depth;

      // Get back to work while we wait for our new graph_task to
      // complete!
      ++current_depth;
      lock.unlock();
      thread_main(graph_task);
      --current_depth;
      --total_depth;

      // The graph task should have completed and the associated future should
      // be marked completed as well since 'thread_main' above is a call
      // blocking an autograd engine thread.
      TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    }
  }
  // graph_task_exec_post_processing is done when the Future is marked as
  // completed in mark_as_completed_and_run_post_processing.
  return graph_task->future_result_;
}
```

1. Initialize thread pool and prepare queue.

      ```
      void Engine::initialize_device_threads_pool() {
      std::call_once(start_device_threads_flag_, &Engine::start_device_threads, this); 

    auto Engine::start_device_threads() -> void {
      //Get num_devices
      for (const auto& impl_atomic : c10::impl::device_guard_impl_registry) {
        auto* impl = impl_atomic.load();
        if (impl) {
          num_devices = std::max(num_devices, impl->deviceCount());
        }
      }

      //----------------------(1)----------------------
      device_ready_queues_ = std::vector<std::shared_ptr<ReadyQueue>>(num_devices);
      for (auto& queue : device_ready_queues_)    {
        // NOLINTNEXTLINE(modernize-make-shared)
        queue.reset(new ReadyQueue());
      }

      //----------------------(2)----------------------
      for (const auto i : c10::irange(num_devices)) {
        std::thread t(&Engine::thread_init, this, i, device_ready_queues_[i], true);
        t.detach();
      }
    ```
    (1). Create ReadyQueue instance for each thread, and put them into device_ready_queue.

    (2). In the for loop, `num_threads` threads are constructed through `std::thread` and they are deliberately run independently through `t.detach()`. In addition, we noticed that `this` pointer was passed in when creating new thread. This pointer points to the current engine instance. Since all threads share the same engine instance, so they can transfer data to each other. 


2. Ready Queue

    CPU ready queue is per GraphTask, but CUDA device ready queues are shared across all graph tasks
    ```
    auto Engine::ready_queue(std::shared_ptr<ReadyQueue> cpu_ready_queue, at::Device device) -> std::shared_ptr<ReadyQueue>{
      if (device.type() == at::kCPU || device.type() == at::DeviceType::Meta) {
        // return the cpu ready queue passed in
        return cpu_ready_queue;
      } else {
        // See Note [Allocating GPUs to autograd threads]
        return device_ready_queues_.at(device.index());
    ```

3. 'thread_main(graph_task)' 

    ```
    auto Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) -> void {

      // local_ready_queue should already been initialized when we get into thread_main
      while (graph_task == nullptr || !graph_task->future_result_->completed()) {
        // local_graph_task represents the graph_task we retrieve from the queue.
        // The outer graph_task represents the overall graph_task we need to execute
        // for reentrant execution.
        std::shared_ptr<GraphTask> local_graph_task;
        {
          // Scope this block of execution since NodeTask is not needed after this
          // block and can be deallocated (release any references to grad tensors
          // as part of inputs_).
          //---------------------------1-------------------------------
          NodeTask task = local_ready_queue->pop();

          if (!(local_graph_task = task.base_.lock())) {
            // GraphTask for function is no longer valid, skipping further
            continue;}

          if (task.fn_ && !local_graph_task->has_error_.load()) {
            AutoGradMode grad_mode(local_graph_task->grad_mode_);
            try {
          //---------------------------2-------------------------------
              evaluate_function(local_graph_task, task.fn_.get(), task.inputs_,  
                                        local_graph_task->cpu_ready_queue_);
            } catch (std::exception& e) {
              thread_on_exception(local_graph_task, task.fn_, e);}}}

          //---------------------------3-------------------------------
        // Decrement the outstanding tasks.
        --local_graph_task->outstanding_tasks_;

          //---------------------------4-------------------------------
        // Check if we've completed execution.
        if (local_graph_task->completed()) {
          local_graph_task->mark_as_completed_and_run_post_processing();

          auto base_owner = local_graph_task->owner_;

          //---------------------------4-------------------------------
          if (worker_device != base_owner) {
            // Synchronize outstanding_tasks_ with queue mutex
            std::atomic_thread_fence(std::memory_order_release);
            ready_queue_by_index(local_graph_task->cpu_ready_queue_, base_owner)
                ->push(NodeTask(local_graph_task, nullptr, InputBuffer(0)));
    
    ```
    1. Continuously take tasks (`NodeTask`) from the `local_ready_queue` in the loop.
    2. `evaluate_function`
        Call `evaluate_function` to execute the NodeTask instance. This function receives NodeTask, the NodeTask saves the derivative calculation function `fn_` of this Node and the gradient before the current Node in the chain.
        ```
        void Engine::evaluate_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputs,
    const std::shared_ptr<ReadyQueue>& cpu_ready_queue) {

  // The InputBuffer::adds that supplied incoming grads took pains to
  // ensure they're safe to consume in the context of the present
  // func's stream (if applicable). So we guard onto that stream
  // before working with the grads in any capacity.
  //---------------------------(1)-------------------------------
  const auto opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);

  // If exec_info_ is not empty, we have to instrument the execution
  auto& exec_info_ = graph_task->exec_info_;
  if (!exec_info_.empty()) {
    auto& fn_info = exec_info_.at(func);
    if (auto* capture_vec = fn_info.captures_.get()) {
      // Lock mutex for writing to graph_task->captured_vars_.
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      for (const auto& capture : *capture_vec) {
        auto& captured_grad = graph_task->captured_vars_[capture.output_idx_];
        captured_grad = inputs[capture.input_idx_];
        for (auto& hook : capture.hooks_) {
  //---------------------------(2)-------------------------------
          captured_grad = (*hook)(captured_grad);
        }
        if (opt_parent_stream) {
          // No need to take graph_task->mutex_ here, we already hold it
          graph_task->leaf_streams.emplace(*opt_parent_stream);
        }
      }
    }
    if (!fn_info.needed_) {
      // Skip execution if we don't need to execute the function.
      return;
    }
  }

  auto outputs = call_function(graph_task, func, inputs);

  auto& fn = *func;
  if (!graph_task->keep_graph_) {
    fn.release_variables();
  }

  int num_outputs = outputs.size();
  if (num_outputs == 0) { // Note: doesn't acquire the mutex
    // Records leaf stream (if applicable)
    // See Note [Streaming backwards]
    if (opt_parent_stream) {
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      graph_task->leaf_streams.emplace(*opt_parent_stream);
    }
    return;
  }

  if (AnomalyMode::is_enabled()) {
    AutoGradMode grad_mode(false);
    for (const auto i : c10::irange(num_outputs)) {
      auto& output = outputs[i];
      at::OptionalDeviceGuard guard(device_of(output));
      if (output.defined() && isnan(output).any().item<uint8_t>()) {
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' returned nan values in its " << i << "th output.";
        throw std::runtime_error(ss.str());
      }
    }
  }

  // Lock mutex for the accesses to GraphTask dependencies_, not_ready_ and cpu_ready_queue_ below
  std::lock_guard<std::mutex> lock(graph_task->mutex_);
  for (const auto i : c10::irange(num_outputs)) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    if (!next.is_valid()) continue;

    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = graph_task->dependencies_;
    auto it = dependencies.find(next.function.get());

    if (it == dependencies.end()) {
      auto name = next.function->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = graph_task->not_ready_;
    auto not_ready_it = not_ready.find(next.function.get());
    if (not_ready_it == not_ready.end()) {
      // Skip functions that aren't supposed to be executed
      if (!exec_info_.empty()) {
        auto it = exec_info_.find(next.function.get());
        if (it == exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
      }
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next.function->num_inputs());

      // Accumulates into buffer
      const auto opt_next_stream = next.function->stream(c10::DeviceType::CUDA);
      input_buffer.add(next.input_nr,
                       std::move(output),
                       opt_parent_stream,
                       opt_next_stream);

      if (is_ready) {
        auto queue = ready_queue(cpu_ready_queue, input_buffer.device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
      } else {
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto &input_buffer = not_ready_it->second;

      // Accumulates into buffer
      const auto opt_next_stream = next.function->stream(c10::DeviceType::CUDA);
      input_buffer.add(next.input_nr,
                       std::move(output),
                       opt_parent_stream,
                       opt_next_stream);
      if (is_ready) {
        auto queue = ready_queue(cpu_ready_queue, input_buffer.device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}
        
        ```

        (1). Function's StreamStream
            A function's stream (for a given device type) is the stream of the first element of its input buffer on a device of that type. If all elements are on the same device they MUST share a stream. If elements are on different devices (across multiple GPUs, for example) they may have different streams.
            ``` 
            c10::optional<c10::Stream> stream(const c10::DeviceType device_type) {
              for (const auto& metadata : input_metadata_) {
                if (metadata.device().type() == device_type) return metadata.stream();}
              return c10::nullopt;}
            ```
            Note for Streaming backwards:
            On CUDA devices the autograd engine's device operations are run on the same stream that ran them in forward. This requires automatically syncing the streams so that function A finishes producing its output before function B consumes it.
            This synchronization occurs when outputs are placed into input buffers. The functions corresponding to input buffer positions have metadata recording their streams from forward, and during backward this data is used to sync the producer's stream with the consumer's.

            When all the inputs of a CUDA function were on the stream used to run this function, or the inputs are on different devices, the function is responsible for properly acquiring them.
            
            See [Stream semantics of backward passes](https://pytorch.org/docs/stable/notes/cuda.html)

            So GraphTask achieves the above semantics by
            a.remembering the current streams on all active CUDA devices in the user-facing thread (aka, the thread that called execute() to launch the GraphTask)
            b.remembering the "leaf streams" (streams each backward leaf node ran on)
            c. during exec_post_processing, for each leaf stream, sync the remembered current streams (on the leaf stream's device) with that leaf stream.
        (2). Hook function: execute a hook function attached with a variable in inputs. (inputs of gradient funciton)

            
    3. Reduce the outstanding_tasks_ of GraphTask corresponding to NodeTask took out in 1
    4. 

        The current worker thread finish the `graph_task`, but the owning thread of the `graph_task` might be sleeping on `pop()` if it does not have work. So we need to send a dummy function task to the owning thread just to ensure that it's not sleeping, so that we can exit the `thread_main`. If it has work, it might see that `graph_task->outstanding_tasks_ == 0` before it gets to the task. NB: This is not necessary if the current thread is the owning thread.

tip:
1. make_shared
  [make_shared V.S. shared_ptr](https://www.jianshu.com/p/03eea8262c11)


2. To understand the [reentrant](https://www.cnblogs.com/clover-toeic/p/3738464.html) backwards problem, we have to notice two aspects of how the autograd engine is implemented today:

    1. When you call Engine::execute(), you want to block until differentiation finishes so that you can get the final result variables of the backwards pass.
    2. The engine operates by having a single worker thread per work queue, and every work queue is pinned to a specific device where the operation is executed.
3. [thread pool](https://wiki.jikexueyuan.com/project/cplusplus-concurrency-action/content/chapter9/9.1-chinese.html)






### Reference

1. [PyTorch Internals 5：Autogradreference](https://zhuanlan.zhihu.com/p/336599887)
2. [autograd源码剖析](https://zhuanlan.zhihu.com/p/336599887)
3. [Pytorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
4. [PyTorch源码浅析(4)：Autograd](https://www.52coding.com.cn/2019/05/05/PyTorch4/#autograd-engine)
5. [A Tour of PyTorch Internals (Part I)](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)




