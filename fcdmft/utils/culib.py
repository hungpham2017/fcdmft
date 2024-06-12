"""
Tools for cuda kernel
"""
import cupy as cp
import numpy as np
from packaging.version import parse
from cupy import cutensor
import cupy_backends.cuda.libs.cutensor as cutensorlib

# Dictionary to map operation names to cuTENSOR operation constants
ops = {
    'IDENTITY': cutensorlib.OP_IDENTITY, 
    'SQRT': cutensorlib.OP_SQRT, 
    'RELU': cutensorlib.OP_RELU,
    'CONJ': cutensorlib.OP_CONJ, 
    'RCP': cutensorlib.OP_RCP, 
    'SIGMOID': cutensorlib.OP_SIGMOID,
    'TANH': cutensorlib.OP_TANH, 
    'EXP': cutensorlib.OP_EXP, 
    'LOG': cutensorlib.OP_LOG,
    'ABS': cutensorlib.OP_ABS, 
    'NEG': cutensorlib.OP_NEG, 
    'SIN': cutensorlib.OP_SIN,
    'COS': cutensorlib.OP_COS, 
    'TAN': cutensorlib.OP_TAN, 
    'SINH': cutensorlib.OP_SINH,
    'COSH': cutensorlib.OP_COSH, 
    'ASIN': cutensorlib.OP_ASIN, 
    'ACOS': cutensorlib.OP_ACOS,
    'ATAN': cutensorlib.OP_ATAN, 
    'ASINH': cutensorlib.OP_ASINH, 
    'ACOSH': cutensorlib.OP_ACOSH,
    'ATANH': cutensorlib.OP_ATANH, 
    'CEIL': cutensorlib.OP_CEIL, 
    'FLOOR': cutensorlib.OP_FLOOR
}

def get_op(op, x):
    """Retrieve the operation from ops based on the operation name and check if the input tensor is complex."""
    if op == 'CONJ' and cp.isrealobj(x):
        return cutensorlib.OP_IDENTITY
    return ops[op]

def contraction(inda, a, indb, b, indc, c=None, alpha=1.0, beta=0.0, opa='IDENTITY', opb='IDENTITY', opc='IDENTITY', dtype=None, buf=None, alg=cutensorlib.ALGO_DEFAULT, wspref=cutensorlib.WORKSPACE_MIN):
    modea = tuple(inda)
    modeb = tuple(indb)
    modec = tuple(indc)

    # Explicitly create modes using cp.cutensor.create_mode
    created_mode_a = cp.cutensor.create_mode(*modea)
    created_mode_b = cp.cutensor.create_mode(*modeb)
    created_mode_c = cp.cutensor.create_mode(*modec)

    if parse(cp.__version__) == parse('13.0.0'):
        a = cp.ascontiguousarray(a)
        b = cp.ascontiguousarray(b)
        
    if c is None:
        dtype = np.result_type(a, b) if dtype is None else dtype
        cshape = [a.shape[modea.index(i)] if i in modea else b.shape[modeb.index(i)] for i in modec]
        if buf is None:
            c = cp.empty(cshape, dtype=dtype)
        else:
            if np.prod(buf.shape) < np.prod(cshape):
                raise MemoryError('Buffer is not large enough, buffer shape:' + str(buf.shape) + " needed shape: " + str(cshape))
            c = cp.ndarray(cshape, dtype=dtype, memptr=buf.data)

    # Perform the contraction with explicitly created modes
    c = cp.cutensor.contraction(alpha, a, created_mode_a, b, created_mode_b, beta, c, created_mode_c, algo=alg, ws_pref=wspref, op_A=get_op(opa, a), op_B=get_op(opb, b), op_C=get_op(opc, c))

    return c
