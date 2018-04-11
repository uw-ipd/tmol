import types

def code(
    argcount,
    kwonlyargcount,
    nlocals,
    stacksize,
    flags,
    codestring,
    constants,
    names,
    varnames,
    filename,
    name,
    firstlineno,
    lnotab,
    freevars,
    cellvars,
    **kwargs
):
    return types.CodeType(
        argcount,
        kwonlyargcount,
        nlocals,
        stacksize,
        flags,
        codestring,
        constants,
        names,
        varnames,
        filename,
        name,
        firstlineno,
        lnotab,
        freevars,
        cellvars
    )

def code_attrs(c):
    cattrs = { n[3:] : getattr(c, n) for n in dir(c) if n.startswith("co_")}
    cattrs["constants"] = cattrs.pop("consts")
    cattrs["codestring"] = cattrs.pop("code")
    return cattrs

def rename_code_object(func, new_name):
    cattrs = code_attrs(func.__code__)
    cattrs["name"] = new_name

    return types.FunctionType(
        code(**cattrs),
        func.__globals__,
        new_name,
        func.__defaults__,
        func.__closure__)
