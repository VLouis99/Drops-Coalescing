# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

"""SWIG wrapper for the CGAL Halfedge Data Structure package provided under the GPL-3.0+ license"""

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _CGAL_HalfedgeDS
else:
    import _CGAL_HalfedgeDS

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
RELATIVE_INDEXING = _CGAL_HalfedgeDS.RELATIVE_INDEXING
ABSOLUTE_INDEXING = _CGAL_HalfedgeDS.ABSOLUTE_INDEXING
class HDS_Halfedge_handle(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HDS_Halfedge_handle_swiginit(self, _CGAL_HalfedgeDS.new_HDS_Halfedge_handle())

    def opposite(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_opposite(self, *args)

    def next(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_next(self, *args)

    def set_next(self, c):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_set_next(self, c)

    def is_border(self):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_is_border(self)

    def prev(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_prev(self, *args)

    def set_prev(self, c):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_set_prev(self, c)

    def vertex(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_vertex(self, *args)

    def set_vertex(self, c):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_set_vertex(self, c)

    def face(self):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_face(self)

    def set_face(self, c):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_set_face(self, c)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle_deepcopy(self, *args)

    def __eq__(self, p):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle___eq__(self, p)

    def __ne__(self, p):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle___ne__(self, p)

    def __hash__(self):
        return _CGAL_HalfedgeDS.HDS_Halfedge_handle___hash__(self)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HDS_Halfedge_handle

# Register HDS_Halfedge_handle in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HDS_Halfedge_handle_swigregister(HDS_Halfedge_handle)

class HDS_Face_handle(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HDS_Face_handle_swiginit(self, _CGAL_HalfedgeDS.new_HDS_Face_handle())

    def halfedge(self, *args):
        return _CGAL_HalfedgeDS.HDS_Face_handle_halfedge(self, *args)

    def set_halfedge(self, c):
        return _CGAL_HalfedgeDS.HDS_Face_handle_set_halfedge(self, c)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HDS_Face_handle_deepcopy(self, *args)

    def __eq__(self, p):
        return _CGAL_HalfedgeDS.HDS_Face_handle___eq__(self, p)

    def __ne__(self, p):
        return _CGAL_HalfedgeDS.HDS_Face_handle___ne__(self, p)

    def __hash__(self):
        return _CGAL_HalfedgeDS.HDS_Face_handle___hash__(self)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HDS_Face_handle

# Register HDS_Face_handle in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HDS_Face_handle_swigregister(HDS_Face_handle)

class HDS_Vertex_handle(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HDS_Vertex_handle_swiginit(self, _CGAL_HalfedgeDS.new_HDS_Vertex_handle())

    def halfedge(self, *args):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle_halfedge(self, *args)

    def set_halfedge(self, c):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle_set_halfedge(self, c)

    def point(self, *args):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle_point(self, *args)

    def set_point(self, p):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle_set_point(self, p)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle_deepcopy(self, *args)

    def __eq__(self, p):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle___eq__(self, p)

    def __ne__(self, p):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle___ne__(self, p)

    def __hash__(self):
        return _CGAL_HalfedgeDS.HDS_Vertex_handle___hash__(self)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HDS_Vertex_handle

# Register HDS_Vertex_handle in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HDS_Vertex_handle_swigregister(HDS_Vertex_handle)

class HDS_Vertex_iterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HDS_Vertex_iterator_swiginit(self, _CGAL_HalfedgeDS.new_HDS_Vertex_iterator())

    def __iter__(self):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator___iter__(self)

    def __next__(self):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator___next__(self)

    def next(self, *args):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator_next(self, *args)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator_deepcopy(self, *args)

    def hasNext(self):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator_hasNext(self)

    def __eq__(self, p):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator___eq__(self, p)

    def __ne__(self, p):
        return _CGAL_HalfedgeDS.HDS_Vertex_iterator___ne__(self, p)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HDS_Vertex_iterator

# Register HDS_Vertex_iterator in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HDS_Vertex_iterator_swigregister(HDS_Vertex_iterator)

class HDS_Halfedge_iterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HDS_Halfedge_iterator_swiginit(self, _CGAL_HalfedgeDS.new_HDS_Halfedge_iterator())

    def __iter__(self):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator___iter__(self)

    def __next__(self):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator___next__(self)

    def next(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator_next(self, *args)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator_deepcopy(self, *args)

    def hasNext(self):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator_hasNext(self)

    def __eq__(self, p):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator___eq__(self, p)

    def __ne__(self, p):
        return _CGAL_HalfedgeDS.HDS_Halfedge_iterator___ne__(self, p)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HDS_Halfedge_iterator

# Register HDS_Halfedge_iterator in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HDS_Halfedge_iterator_swigregister(HDS_Halfedge_iterator)

class HDS_Face_iterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HDS_Face_iterator_swiginit(self, _CGAL_HalfedgeDS.new_HDS_Face_iterator())

    def __iter__(self):
        return _CGAL_HalfedgeDS.HDS_Face_iterator___iter__(self)

    def __next__(self):
        return _CGAL_HalfedgeDS.HDS_Face_iterator___next__(self)

    def next(self, *args):
        return _CGAL_HalfedgeDS.HDS_Face_iterator_next(self, *args)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HDS_Face_iterator_deepcopy(self, *args)

    def hasNext(self):
        return _CGAL_HalfedgeDS.HDS_Face_iterator_hasNext(self)

    def __eq__(self, p):
        return _CGAL_HalfedgeDS.HDS_Face_iterator___eq__(self, p)

    def __ne__(self, p):
        return _CGAL_HalfedgeDS.HDS_Face_iterator___ne__(self, p)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HDS_Face_iterator

# Register HDS_Face_iterator in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HDS_Face_iterator_swigregister(HDS_Face_iterator)

class HalfedgeDS_modifier(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _CGAL_HalfedgeDS.HalfedgeDS_modifier_swiginit(self, _CGAL_HalfedgeDS.new_HalfedgeDS_modifier())

    def begin_surface(self, v, f, h=0, mode=RELATIVE_INDEXING):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_begin_surface(self, v, f, h, mode)

    def end_surface(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_end_surface(self)

    def add_vertex(self, p):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_add_vertex(self, p)

    def begin_facet(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_begin_facet(self)

    def end_facet(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_end_facet(self)

    def add_vertex_to_facet(self, i):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_add_vertex_to_facet(self, i)

    def rollback(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_rollback(self)

    def clear(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_modifier_clear(self)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HalfedgeDS_modifier

# Register HalfedgeDS_modifier in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HalfedgeDS_modifier_swigregister(HalfedgeDS_modifier)

class HalfedgeDS(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _CGAL_HalfedgeDS.HalfedgeDS_swiginit(self, _CGAL_HalfedgeDS.new_HalfedgeDS(*args))

    def reserve(self, c1, c2, c3):
        return _CGAL_HalfedgeDS.HalfedgeDS_reserve(self, c1, c2, c3)

    def size_of_vertices(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_size_of_vertices(self)

    def size_of_halfedges(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_size_of_halfedges(self)

    def size_of_faces(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_size_of_faces(self)

    def capacity_of_vertices(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_capacity_of_vertices(self)

    def capacity_of_halfedges(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_capacity_of_halfedges(self)

    def capacity_of_faces(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_capacity_of_faces(self)

    def bytes(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_bytes(self)

    def bytes_reserved(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_bytes_reserved(self)

    def vertices(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_vertices(self)

    def halfedges(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_halfedges(self)

    def faces(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_faces(self)

    def vertices_push_back(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_vertices_push_back(self, *args)

    def edges_push_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_edges_push_back(self)

    def faces_push_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_faces_push_back(self)

    def vertices_pop_front(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_vertices_pop_front(self)

    def vertices_pop_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_vertices_pop_back(self)

    def vertices_erase(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_vertices_erase(self, *args)

    def edges_pop_front(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_edges_pop_front(self)

    def edges_pop_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_edges_pop_back(self)

    def edges_erase(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_edges_erase(self, *args)

    def faces_pop_front(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_faces_pop_front(self)

    def faces_pop_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_faces_pop_back(self)

    def faces_erase(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_faces_erase(self, *args)

    def vertices_clear(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_vertices_clear(self)

    def edges_clear(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_edges_clear(self)

    def faces_clear(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_faces_clear(self)

    def clear(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_clear(self)

    def normalize_border(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_normalize_border(self)

    def size_of_border_halfedges(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_size_of_border_halfedges(self)

    def size_of_border_edges(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_size_of_border_edges(self)

    def border_halfedges(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_border_halfedges(self)

    def deepcopy(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_deepcopy(self, *args)

    def delegate(self, modifier):
        return _CGAL_HalfedgeDS.HalfedgeDS_delegate(self, modifier)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HalfedgeDS

# Register HalfedgeDS in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HalfedgeDS_swigregister(HalfedgeDS)

class HalfedgeDS_decorator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, hds):
        _CGAL_HalfedgeDS.HalfedgeDS_decorator_swiginit(self, _CGAL_HalfedgeDS.new_HalfedgeDS_decorator(hds))

    def vertices_push_back(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_vertices_push_back(self, *args)

    def faces_push_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_faces_push_back(self)

    def create_loop(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_create_loop(self)

    def create_segment(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_create_segment(self)

    def vertices_pop_front(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_vertices_pop_front(self)

    def vertices_pop_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_vertices_pop_back(self)

    def vertices_erase(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_vertices_erase(self, *args)

    def faces_pop_front(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_faces_pop_front(self)

    def faces_pop_back(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_faces_pop_back(self)

    def faces_erase(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_faces_erase(self, *args)

    def erase_face(self, c):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_erase_face(self, c)

    def erase_connected_component(self, c):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_erase_connected_component(self, c)

    def keep_largest_connected_components(self, c):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_keep_largest_connected_components(self, c)

    def make_hole(self, c):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_make_hole(self, c)

    def fill_hole(self, c):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_fill_hole(self, c)

    def add_face_to_border(self, c1, c2):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_add_face_to_border(self, c1, c2)

    def split_face(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_split_face(self, *args)

    def join_face(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_join_face(self, *args)

    def split_vertex(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_split_vertex(self, *args)

    def join_vertex(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_join_vertex(self, *args)

    def create_center_vertex(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_create_center_vertex(self, *args)

    def erase_center_vertex(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_erase_center_vertex(self, *args)

    def split_loop(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_split_loop(self, *args)

    def join_loop(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_join_loop(self, *args)

    def is_valid(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_is_valid(self, *args)

    def normalized_border_is_valid(self, *args):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_normalized_border_is_valid(self, *args)

    def inside_out(self):
        return _CGAL_HalfedgeDS.HalfedgeDS_decorator_inside_out(self)
    __swig_destroy__ = _CGAL_HalfedgeDS.delete_HalfedgeDS_decorator

# Register HalfedgeDS_decorator in _CGAL_HalfedgeDS:
_CGAL_HalfedgeDS.HalfedgeDS_decorator_swigregister(HalfedgeDS_decorator)



