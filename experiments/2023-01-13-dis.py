import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        """
    # Disassembling Python Bytecode

    Python is an interpreted language, meaning that programs written in Python aren't compiled to assembly before being executed, and instead a program called an interpreter executes a sequence of higher-level instructions, represented in Python Bytecode.
    The abstract model of the Python virtual machine is a stack machine, where the operands to instructions are loaded from an abstract stack, and results are pushed onto the same stack.
    Python provides an API to inspect these instructions and manipulate them in the [`dis`](https://docs.python.org/3/library/dis.html) module.
    This post is a short primer on understanding Python Bytecode, and how it can be manipulated to optimise a Python function.
    """
    )
    return


@app.cell
def __(measure):
    def zero():
        return 0

    measure("zero()")
    return zero,


@app.cell
def __(Bytecode, show, zero):
    def times_zero(x: int) -> int:
        return x * 0

    dis_times_zero = Bytecode(zero)
    show(dis_times_zero)
    return dis_times_zero, times_zero


@app.cell
def __(measure):
    def times_two_0(x: int) -> bool:
        return x * 2

    measure("times_two_0(100)")
    return times_two_0,


@app.cell
def __(measure):
    def times_two_1(x: int) -> bool:
        return x + x

    measure("times_two_1(100)")
    return times_two_1,


@app.cell
def __(Bytecode, show, times_two_0):
    dis_times_two_0 = Bytecode(times_two_0)
    show(dis_times_two_0)
    return dis_times_two_0,


@app.cell
def __(dis, times_two_1):
    dis(times_two_1)
    return


@app.cell
def __(dis, times_two_0):
    def double_list(xs: list[int]) -> list[int]:
        return [times_two_0(x) for x in xs]

    dis_double_list = dis(double_list)
    return dis_double_list, double_list


@app.cell
def __(dis_double_list):
    type(dis_double_list)
    return


@app.cell
def __(mo):
    from dis import Bytecode

    def show(b: Bytecode) -> mo.Html:
        return mo.md(
            f"""
    ```
    {b.dis()}
    ```
    """
        )
    return Bytecode, show


@app.cell
def __():
    from timeit import timeit

    def measure(code: str) -> str:
        ms = timeit(code, number=1000, globals=globals())
        return f"{ms:.4e}ms"
    return measure, timeit


@app.cell
def __():
    import datetime

    td = datetime.timedelta(milliseconds=0.023)
    return datetime, td


@app.cell
def __():
    a = 4
    return a,


@app.cell
def __(mo):
    bla = mo.ui.slider(1, 10)
    bla
    return bla,


@app.cell
def __():
    b = 2
    return b,


@app.cell
def __(a, bla):
    c = a + bla.value
    c
    return c,


if __name__ == "__main__":
    app.run()
