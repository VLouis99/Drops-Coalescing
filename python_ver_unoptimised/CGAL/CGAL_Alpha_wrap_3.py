# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

"""SWIG wrapper for the CGAL 3D Alpha Wrapping provided under the GPL-3.0+ license"""

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _CGAL_Alpha_wrap_3
else:
    import _CGAL_Alpha_wrap_3

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


import CGAL.CGAL_Kernel
import CGAL.CGAL_Polyhedron_3
class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _CGAL_Alpha_wrap_3.delete_SwigPyIterator

    def value(self):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_equal(self, x)

    def copy(self):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_copy(self)

    def next(self):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_next(self)

    def __next__(self):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___next__(self)

    def previous(self):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_previous(self)

    def advance(self, n):
        return _CGAL_Alpha_wrap_3.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _CGAL_Alpha_wrap_3.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _CGAL_Alpha_wrap_3:
_CGAL_Alpha_wrap_3.SwigPyIterator_swigregister(SwigPyIterator)

class Point_3_Vector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___nonzero__(self)

    def __bool__(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___bool__(self)

    def __len__(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___len__(self)

    def __getslice__(self, i, j):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector___setitem__(self, *args)

    def pop(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_pop(self)

    def append(self, x):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_append(self, x)

    def empty(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_empty(self)

    def size(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_size(self)

    def swap(self, v):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_swap(self, v)

    def begin(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_begin(self)

    def end(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_end(self)

    def rbegin(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_rbegin(self)

    def rend(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_rend(self)

    def clear(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_clear(self)

    def get_allocator(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_get_allocator(self)

    def pop_back(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_pop_back(self)

    def erase(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_erase(self, *args)

    def __init__(self, *args):
        _CGAL_Alpha_wrap_3.Point_3_Vector_swiginit(self, _CGAL_Alpha_wrap_3.new_Point_3_Vector(*args))

    def push_back(self, x):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_push_back(self, x)

    def front(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_front(self)

    def back(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_back(self)

    def assign(self, n, x):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_assign(self, n, x)

    def resize(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_resize(self, *args)

    def insert(self, *args):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_insert(self, *args)

    def reserve(self, n):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_reserve(self, n)

    def capacity(self):
        return _CGAL_Alpha_wrap_3.Point_3_Vector_capacity(self)
    __swig_destroy__ = _CGAL_Alpha_wrap_3.delete_Point_3_Vector

# Register Point_3_Vector in _CGAL_Alpha_wrap_3:
_CGAL_Alpha_wrap_3.Point_3_Vector_swigregister(Point_3_Vector)

class Int_Vector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CGAL_Alpha_wrap_3.Int_Vector___nonzero__(self)

    def __bool__(self):
        return _CGAL_Alpha_wrap_3.Int_Vector___bool__(self)

    def __len__(self):
        return _CGAL_Alpha_wrap_3.Int_Vector___len__(self)

    def __getslice__(self, i, j):
        return _CGAL_Alpha_wrap_3.Int_Vector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CGAL_Alpha_wrap_3.Int_Vector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector___setitem__(self, *args)

    def pop(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_pop(self)

    def append(self, x):
        return _CGAL_Alpha_wrap_3.Int_Vector_append(self, x)

    def empty(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_empty(self)

    def size(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_size(self)

    def swap(self, v):
        return _CGAL_Alpha_wrap_3.Int_Vector_swap(self, v)

    def begin(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_begin(self)

    def end(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_end(self)

    def rbegin(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_rbegin(self)

    def rend(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_rend(self)

    def clear(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_clear(self)

    def get_allocator(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_get_allocator(self)

    def pop_back(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_pop_back(self)

    def erase(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector_erase(self, *args)

    def __init__(self, *args):
        _CGAL_Alpha_wrap_3.Int_Vector_swiginit(self, _CGAL_Alpha_wrap_3.new_Int_Vector(*args))

    def push_back(self, x):
        return _CGAL_Alpha_wrap_3.Int_Vector_push_back(self, x)

    def front(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_front(self)

    def back(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_back(self)

    def assign(self, n, x):
        return _CGAL_Alpha_wrap_3.Int_Vector_assign(self, n, x)

    def resize(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector_resize(self, *args)

    def insert(self, *args):
        return _CGAL_Alpha_wrap_3.Int_Vector_insert(self, *args)

    def reserve(self, n):
        return _CGAL_Alpha_wrap_3.Int_Vector_reserve(self, n)

    def capacity(self):
        return _CGAL_Alpha_wrap_3.Int_Vector_capacity(self)
    __swig_destroy__ = _CGAL_Alpha_wrap_3.delete_Int_Vector

# Register Int_Vector in _CGAL_Alpha_wrap_3:
_CGAL_Alpha_wrap_3.Int_Vector_swigregister(Int_Vector)

class Polygon_Vector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___nonzero__(self)

    def __bool__(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___bool__(self)

    def __len__(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___len__(self)

    def __getslice__(self, i, j):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector___setitem__(self, *args)

    def pop(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_pop(self)

    def append(self, x):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_append(self, x)

    def empty(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_empty(self)

    def size(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_size(self)

    def swap(self, v):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_swap(self, v)

    def begin(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_begin(self)

    def end(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_end(self)

    def rbegin(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_rbegin(self)

    def rend(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_rend(self)

    def clear(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_clear(self)

    def get_allocator(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_get_allocator(self)

    def pop_back(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_pop_back(self)

    def erase(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_erase(self, *args)

    def __init__(self, *args):
        _CGAL_Alpha_wrap_3.Polygon_Vector_swiginit(self, _CGAL_Alpha_wrap_3.new_Polygon_Vector(*args))

    def push_back(self, x):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_push_back(self, x)

    def front(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_front(self)

    def back(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_back(self)

    def assign(self, n, x):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_assign(self, n, x)

    def resize(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_resize(self, *args)

    def insert(self, *args):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_insert(self, *args)

    def reserve(self, n):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_reserve(self, n)

    def capacity(self):
        return _CGAL_Alpha_wrap_3.Polygon_Vector_capacity(self)
    __swig_destroy__ = _CGAL_Alpha_wrap_3.delete_Polygon_Vector

# Register Polygon_Vector in _CGAL_Alpha_wrap_3:
_CGAL_Alpha_wrap_3.Polygon_Vector_swigregister(Polygon_Vector)


def alpha_wrap_3(*args):
    return _CGAL_Alpha_wrap_3.alpha_wrap_3(*args)


