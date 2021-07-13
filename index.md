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
Now in Now in THPEngine_run_backward, we can find out what backward() exactly do.
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
        auto gradient_edge = torch::autograd::impl::gradient_edge(variable);
        roots.push_back(std::move(gradient_edge));

```
1. What is edge_list
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
    In Pytorch graph, each edge stores the function (function) that the edge points at, and it also stores index(input_nr) of this function, while function could have more than one input, i.e. input_nr indicates that this edge is which input to the function.

2. gradient_edge
If grad_fn is null (as is the case for a leaf node), it instead interpret the gradient function to be a gradient accumulator, which will accumulate its inputs into the grad property of the variable. Note that only variables which have `requires_grad = True` can have gradient accumulators.

```
Edge gradient_edge(const Variable& self???) {
    if (const auto& gradient = self.grad_fn()) {
      return Edge(gradient, self.output_nr());
    } else {
      return Edge(grad_accumulator(self), 0);
    }
  }
```


For each tensor, 











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
