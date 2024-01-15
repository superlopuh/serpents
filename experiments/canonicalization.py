import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __():
    from xdsl.dialects import riscv, riscv_func, builtin
    from xdsl.builder import ImplicitBuilder, Builder
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.dead_code_elimination import dce
    from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, GreedyRewritePatternApplier
    from xdsl.ir import MLContext
    return (
        Builder,
        CanonicalizePass,
        GreedyRewritePatternApplier,
        ImplicitBuilder,
        MLContext,
        PatternRewriteWalker,
        PatternRewriter,
        builtin,
        dce,
        riscv,
        riscv_func,
    )


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    a_val = mo.ui.slider(-10, 10)
    a_val
    return a_val,


@app.cell
def __(
    Builder,
    CanonicalizePass,
    MLContext,
    a_val,
    b_val,
    builtin,
    riscv,
    riscv_func,
):
    @builtin.ModuleOp
    @Builder.implicit_region
    def module():
        @Builder.implicit_region
        def body():
            a = riscv.LiOp(a_val.value).rd
            b = riscv.LiOp(b_val.value).rd
            c = riscv.AddOp(a, b, rd=riscv.IntRegisterType.unallocated()).rd
            res = riscv.MVOp(c, rd=riscv.IntRegisterType.a_register(0)).rd
            riscv_func.ReturnOp(res)
        riscv_func.FuncOp("main", body, ((), ()))



    new = module.clone()
    ctx = MLContext()
    ctx.load_dialect(riscv.RISCV)
    CanonicalizePass().apply(ctx, new)
    str(module) + "\n" + str(new)
    return ctx, module, new


@app.cell
def __(mo):
    b_val = mo.ui.slider(-10, 10)
    b_val
    return b_val,


@app.cell
def __(new):
    from xdsl.interpreter import Interpreter
    from xdsl.interpreters.riscv import RiscvFunctions
    from xdsl.interpreters.riscv_func import RiscvFuncFunctions

    i = Interpreter(new)
    i.register_implementations(RiscvFunctions())
    i.register_implementations(RiscvFuncFunctions())

    res = i.call_op("main", ())

    res
    return Interpreter, RiscvFuncFunctions, RiscvFunctions, i, res


if __name__ == "__main__":
    app.run()
