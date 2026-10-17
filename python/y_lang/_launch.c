/* TorchKernel.launch's native path: the kernel arguments converted and
 * cuLaunchKernel called in one C call.
 *
 * Through ctypes, converting cuLaunchKernel's eleven arguments cost about
 * 0.9 us of the 2.9 us a launch took, and setting the parameter slots one
 * ctypes attribute at a time most of the rest; the driver's own call is about
 * 1 us. y_lang.torch_interop compiles this file with the system C compiler on
 * first use, caches the module, and launches through ctypes as before when it
 * cannot.
 *
 * Each argument becomes one 8-byte parameter slot exactly as the ctypes path
 * fills it: an object with a callable `data_ptr` gives that pointer; an int
 * its value modulo 2**64; a float its IEEE single bits (OverflowError where
 * `struct.pack("<f", x)` raises one); anything else is a TypeError.
 * `pack` returns the slots, so a test can compare the two paths byte for
 * byte. cuLaunchKernel is not linked: `set_driver` receives the address
 * ctypes resolved in libcuda.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

typedef int (*cu_launch_kernel_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                  unsigned, void *, void **, void **);

static cu_launch_kernel_t cu_launch_kernel;
static PyObject *data_ptr_name;

#define INLINE_SLOTS 32

/* `getattr(x, "data_ptr", None)`: a new reference, NULL with no error when
 * the attribute is missing, NULL with an error when looking it up raised
 * anything else. */
static PyObject *optional_data_ptr(PyObject *x) {
#if PY_VERSION_HEX >= 0x030D0000
    PyObject *m = NULL;
    if (PyObject_GetOptionalAttr(x, data_ptr_name, &m) < 0)
        return NULL;
    return m;
#else
    PyObject *m = PyObject_GetAttr(x, data_ptr_name);
    if (m == NULL && PyErr_ExceptionMatches(PyExc_AttributeError))
        PyErr_Clear();
    return m;
#endif
}

/* Fill `slot` from `x`; 0, or -1 with an exception set. */
static int fill_slot(PyObject *x, uint64_t *slot) {
    /* An exact int or float has no `data_ptr`; skip the lookup. */
    if (!PyLong_CheckExact(x) && !PyFloat_CheckExact(x)) {
        PyObject *m = optional_data_ptr(x);
        if (m == NULL && PyErr_Occurred())
            return -1;
        if (m != NULL) {
            int callable = PyCallable_Check(m);
            if (callable) {
                PyObject *r = PyObject_CallObject(m, NULL);
                Py_DECREF(m);
                if (r == NULL)
                    return -1;
                unsigned long long v = PyLong_AsUnsignedLongLongMask(r);
                Py_DECREF(r);
                if (v == (unsigned long long)-1 && PyErr_Occurred())
                    return -1;
                *slot = v;
                return 0;
            }
            Py_DECREF(m);
        }
    }
    if (PyLong_Check(x)) {
        unsigned long long v = PyLong_AsUnsignedLongLongMask(x);
        if (v == (unsigned long long)-1 && PyErr_Occurred())
            return -1;
        *slot = v;
        return 0;
    }
    if (PyFloat_Check(x)) {
        unsigned char b[4];
#if PY_VERSION_HEX >= 0x030B0000
        if (PyFloat_Pack4(PyFloat_AS_DOUBLE(x), (char *)b, 1) < 0)
#else
        if (_PyFloat_Pack4(PyFloat_AS_DOUBLE(x), b, 1) < 0)
#endif
            return -1;
        *slot = (uint64_t)b[0] | (uint64_t)b[1] << 8 | (uint64_t)b[2] << 16 | (uint64_t)b[3] << 24;
        return 0;
    }
    PyErr_Format(PyExc_TypeError, "Unsupported argument type: %R", (PyObject *)Py_TYPE(x));
    return -1;
}

/* The slots for the tuple `args`, in `inline_slots` when they fit and in
 * memory the caller frees otherwise; NULL with an exception set on failure.
 * A tuple, not the caller's list: `data_ptr` runs Python code, which could
 * resize a list under the loop. */
static uint64_t *fill_slots(PyObject *args, uint64_t *inline_slots, Py_ssize_t *count) {
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    PyObject **items = &PyTuple_GET_ITEM(args, 0);
    uint64_t *slots = inline_slots;
    if (n > INLINE_SLOTS) {
        slots = PyMem_Malloc((size_t)n * sizeof *slots);
        if (slots == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        if (fill_slot(items[i], &slots[i]) < 0) {
            if (slots != inline_slots)
                PyMem_Free(slots);
            return NULL;
        }
    }
    *count = n;
    return slots;
}

static PyObject *set_driver(PyObject *self, PyObject *address) {
    void *p = PyLong_AsVoidPtr(address);
    if (p == NULL) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "cuLaunchKernel's address is NULL");
        return NULL;
    }
    cu_launch_kernel = (cu_launch_kernel_t)p;
    Py_RETURN_NONE;
}

/* A pointer-sized handle: None is NULL, as for a ctypes c_void_p. */
static int as_handle(PyObject *x, void **out) {
    if (x == Py_None) {
        *out = NULL;
        return 0;
    }
    *out = PyLong_AsVoidPtr(x);
    return *out == NULL && PyErr_Occurred() ? -1 : 0;
}

/* launch(function, gx, gy, gz, bx, by, bz, shared_bytes, stream, args) ->
 * cuLaunchKernel's result code. */
static PyObject *launch(PyObject *self, PyObject *const *a, Py_ssize_t nargs) {
    if (nargs != 10) {
        PyErr_Format(PyExc_TypeError, "launch() takes 10 arguments (%zd given)", nargs);
        return NULL;
    }
    if (cu_launch_kernel == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "set_driver() was not called");
        return NULL;
    }
    void *function, *stream;
    unsigned dims[7];
    if (as_handle(a[0], &function) < 0)
        return NULL;
    for (int i = 0; i < 7; i++) {
        /* A ctypes c_uint argument: an int modulo 2**32. */
        unsigned long v = PyLong_AsUnsignedLongMask(a[1 + i]);
        if (v == (unsigned long)-1 && PyErr_Occurred())
            return NULL;
        dims[i] = (unsigned)v;
    }
    if (as_handle(a[8], &stream) < 0)
        return NULL;
    PyObject *seq = PySequence_Tuple(a[9]);
    if (seq == NULL)
        return NULL;
    uint64_t inline_slots[INLINE_SLOTS];
    void *inline_params[INLINE_SLOTS];
    Py_ssize_t n = 0;
    uint64_t *slots = fill_slots(seq, inline_slots, &n);
    Py_DECREF(seq);
    if (slots == NULL)
        return NULL;
    void **params = inline_params;
    if (n > INLINE_SLOTS) {
        params = PyMem_Malloc((size_t)n * sizeof *params);
        if (params == NULL) {
            PyMem_Free(slots);
            return PyErr_NoMemory();
        }
    }
    for (Py_ssize_t i = 0; i < n; i++)
        params[i] = &slots[i];
    int result;
    /* ctypes released the GIL around the call; a launch that waits for room
     * in the launch queue does not hold up other threads here either. */
    Py_BEGIN_ALLOW_THREADS
    result = cu_launch_kernel(function, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], stream,
                              params, NULL);
    Py_END_ALLOW_THREADS
    if (slots != inline_slots) {
        PyMem_Free(slots);
        PyMem_Free(params);
    }
    return PyLong_FromLong(result);
}

/* pack(args) -> the parameter slots `launch` would pass, as bytes. */
static PyObject *pack(PyObject *self, PyObject *args) {
    PyObject *seq = PySequence_Tuple(args);
    if (seq == NULL)
        return NULL;
    uint64_t inline_slots[INLINE_SLOTS];
    Py_ssize_t n = 0;
    uint64_t *slots = fill_slots(seq, inline_slots, &n);
    Py_DECREF(seq);
    if (slots == NULL)
        return NULL;
    PyObject *out = PyBytes_FromStringAndSize((const char *)slots, n * (Py_ssize_t)sizeof *slots);
    if (slots != inline_slots)
        PyMem_Free(slots);
    return out;
}

static PyMethodDef methods[] = {
    {"set_driver", set_driver, METH_O, "set_driver(address): cuLaunchKernel's address in libcuda."},
    {"launch", (PyCFunction)(void (*)(void))launch, METH_FASTCALL,
     "launch(function, gx, gy, gz, bx, by, bz, shared_bytes, stream, args) -> CUresult"},
    {"pack", pack, METH_O, "pack(args) -> bytes: the parameter slots launch() would pass."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ylaunch", NULL, -1, methods};

PyMODINIT_FUNC PyInit__ylaunch(void) {
    data_ptr_name = PyUnicode_InternFromString("data_ptr");
    if (data_ptr_name == NULL)
        return NULL;
    return PyModule_Create(&module);
}
