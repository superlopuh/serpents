import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("""
    # Compiling PyTorch to RISC-V

    ## Part 0: Converting PyTorch to ONNX

    This is a first of a series of blog posts on how to use [xDSL](https://xdsl.dev/) to compile a neural network implemented in [PyTorch](https://pytorch.org/) to an AI accelerator built on top of the [RISC-V ISA](https://riscv.org/).
    The objective of these blog posts is to serve as an introduction to writing compilers in general, and to discuss some of the interesting challenges of compiling neural networks specifically.

    As a starting point, this post will follow [this tutorial in the PyTorch documentation](https://pytorch.org/tutorials//beginner/onnx/export_simple_model_to_onnx_tutorial.html) on how to convert a PyTorch Model to ONNX.
    """)
    return


@app.cell
def __(mo):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class MyModel(nn.Module):

        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    mo.md("""
    ### 1. Author a simple image classifier model

    ``` python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class MyModel(nn.Module):

        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    ```
    """)
    return F, MyModel, nn, torch


@app.cell
def __(MyModel, mo, torch):
    torch.manual_seed(0)
    torch_model = MyModel()
    torch_input = torch.randn(1, 1, 32, 32)
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)

    mo.md("""
    ### 2. Export the model to ONNX format

    ``` python
    torch.manual_seed(0)
    torch_model = MyModel()
    torch_input = torch.randn(1, 1, 32, 32)
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    ```
    """)
    return onnx_program, torch_input, torch_model


@app.cell
def __(mo, onnx_program):
    onnx_program.save("my_image_classifier.onnx")

    mo.md("""
    ### 3. Save the ONNX model in a file

    ``` python
    onnx_program.save("my_image_classifier.onnx")
    ```
    """)
    return


@app.cell
def __(mo, onnx_program, torch_input):
    import onnxruntime

    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    print(f"Input length: {len(onnx_input)}")
    print(f"Sample input: {onnx_input}")

    ort_session = onnxruntime.InferenceSession("./my_image_classifier.onnx", providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

    mo.md("""
    ### 4. Execute the ONNX model with ONNX Runtime

    ``` python
    import onnxruntime

    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
    print(f"Input length: {len(onnx_input)}")
    print(f"Sample input: {onnx_input}")

    ort_session = onnxruntime.InferenceSession("./my_image_classifier.onnx", providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    ```
    """)
    return (
        onnx_input,
        onnxruntime,
        onnxruntime_input,
        onnxruntime_outputs,
        ort_session,
        to_numpy,
    )


@app.cell
def __(
    mo,
    onnx_program,
    onnxruntime_outputs,
    torch,
    torch_input,
    torch_model,
):
    torch_outputs = torch_model(torch_input)
    torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Sample output: {onnxruntime_outputs}")

    mo.md("""
    ### 5. Compare the PyTorch results with the ones from the ONNX Runtime

    ``` python
    torch_outputs = torch_model(torch_input)
    torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Sample output: {onnxruntime_outputs}")
    ```
    """)
    return onnxruntime_output, torch_output, torch_outputs


if __name__ == "__main__":
    app.run()
