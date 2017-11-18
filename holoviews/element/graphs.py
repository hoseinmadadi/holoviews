from types import FunctionType

import param
import numpy as np

from ..core import Dimension, Dataset, Element2D
from ..core.dimension import redim
from ..core.util import max_range
from ..core.operation import Operation
from .chart import Points
from .path import Path
from .util import split_path, pd

try:
    from datashader.layout import LayoutAlgorithm as ds_layout
except:
    ds_layout = None


class redim_graph(redim):
    """
    Extension for the redim utility that allows re-dimensioning
    Graph objects including their nodes and edgepaths.
    """

    def __call__(self, specs=None, **dimensions):
        redimmed = super(redim_graph, self).__call__(specs, **dimensions)
        new_data = (redimmed.data,)
        if self.parent.nodes:
            new_data = new_data + (self.parent.nodes.redim(specs, **dimensions),)
        if self.parent._edgepaths:
            new_data = new_data + (self.parent.edgepaths.redim(specs, **dimensions),)
        return redimmed.clone(new_data)


def circular_layout(nodes):
    N = len(nodes)
    circ = np.pi/N*np.arange(N)*2
    x = np.cos(circ)
    y = np.sin(circ)
    return (x, y, nodes)


def cubic_bezier(start, end, control=(0, 0), steps=np.linspace(0, 1, 100)):
    sx, sy = start
    ex, ey = end
    cx, cy = control
    xs = (1-steps)**2*sx + 2*(1-steps)*steps*cx+steps**2*ex
    ys = (1-steps)**2*sy + 2*(1-steps)*steps*cy+steps**2*ey
    return np.column_stack([xs, ys])


def quadratic_bezier(start, end, c0=(0, 0), c1=(0, 0), steps=np.linspace(0, 1, 100)):
    sx, sy = start
    ex, ey = end
    cx0, cy0 = c0
    cx1, cy1 = c1
    xs = (1-steps)**3*sx + 3*((1-steps)**2)*steps*cx0 + 3*(1-steps)*steps**2*cx1 + steps**3*ex
    ys = (1-steps)**3*sy + 3*((1-steps)**2)*steps*cy0 + 3*(1-steps)*steps**2*cy1 + steps**3*ey
    return np.column_stack([xs, ys])


def direct(start, end):
    sx, sy = start
    ex, ey = end
    return np.array([(sx, sy), (ex, ey)])


def connect_edges_pd(graph, edge_type='direct'):
    edges = graph.dframe()
    edges.index.name = 'graph_edge_index'
    edges = edges.reset_index()
    nodes = graph.nodes.dframe()
    src, tgt = graph.kdims
    x, y, idx = graph.nodes.kdims[:3]

    df = pd.merge(edges, nodes, left_on=[src.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'src_x', y.name: 'src_y'})

    df = pd.merge(df, nodes, left_on=[tgt.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'dst_x', y.name: 'dst_y'})
    df = df.sort_values('graph_edge_index')

    edge_segments = []
    N = len(nodes)
    for i, edge in df.iterrows():
        start = edge['src_x'], edge['src_y']
        end = edge['dst_x'], edge['dst_y']
        if edge_type == 'direct':
            segment = direct(start, end)
        elif edge_type == 'cubic_bezier':
            segment = cubic_bezier(start, end)
        edge_segments.append(segment)
    return edge_segments


def connect_edges(graph, edge_type='direct'):
    paths = []
    for start, end in graph.array(self.kdims):
        start_ds = graph.nodes[:, :, start]
        end_ds = graph.nodes[:, :, end]
        if not len(start_ds) or not len(end_ds):
            raise ValueError('Could not find node positions for all edges')
        start = start_ds.array(start_ds.kdims[:2]).T
        end = end_ds.array(end_ds.kdims[:2]).T
        if edge_type == 'direct':
            segment = direct(start, end)
        elif edge_type == 'cubic_bezier':
            segment = cubic_bezier(start, end)
        paths.append(segment)
    return paths


def search_indices(values, source):
    orig_indices = source.argsort()
    return orig_indices[np.searchsorted(source[orig_indices], values)]


def compute_chords(element):
    source = element.dimension_values(0, expanded=False)
    target = element.dimension_values(1, expanded=False)
    nodes = np.unique(np.concatenate([source, target]))

    src, tgt = (element.dimension_values(i) for i in range(2))
    src_idx = search_indices(src, nodes)
    tgt_idx = search_indices(tgt, nodes)
    if element.vdims:
        values = element.dimension_values(2)
    else:
        values = np.ones(len(element))

    matrix = np.zeros((len(nodes), len(nodes)))
    for s, t, v in zip(src_idx, tgt_idx, values):
        matrix[s, t] = v

    weights_of_areas = (matrix.sum(axis=0) + matrix.sum(axis=1)) - matrix.diagonal()
    areas_in_radians = (weights_of_areas / weights_of_areas.sum()) * (2 * np.pi)

    # We add a zero in the begging for the cumulative sum
    points = np.zeros((areas_in_radians.shape[0] + 1))
    points[1:] = areas_in_radians
    points = points.cumsum()

    # Compute edge points
    xs = np.cos(points)
    ys = np.sin(points)

    # Compute mid-points
    midpoints = np.convolve(points, [0.5, 0.5],
                            mode='valid')
    mxs = np.cos(midpoints)
    mys = np.sin(midpoints)

    all_areas = []
    for i in range(areas_in_radians.shape[0]):
        n_conn = weights_of_areas[i]
        p0, p1 = points[i], points[i+1]
        angles = np.linspace(p0, p1, n_conn)
        coords = list(zip(np.cos(angles), np.sin(angles)))
        all_areas.append(coords)

    empty = np.array([[np.NaN, np.NaN]])
    paths = []
    for i in range(len(element)):
        src_area, tgt_area = all_areas[src_idx[i]], all_areas[tgt_idx[i]]
        subpaths = []
        for _ in range(int(values[i])):
            x0, y0 = src_area.pop()
            x1, y1 = tgt_area.pop()
            b = quadratic_bezier((x0, y0), (x1, y1), (x0/2., y0/2.),
                                 (x1/2., y1/2.))
            subpaths.append(b)
            subpaths.append(empty)
        if subpaths:
            paths.append(np.concatenate(subpaths))

    return (mxs, mys, nodes), paths


class layout_nodes(Operation):
    """
    Accepts a Graph and lays out the corresponding nodes with the
    supplied networkx layout function. If no layout function is
    supplied uses a simple circular_layout function. Also supports
    LayoutAlgorithm function provided in datashader layouts.
    """

    only_nodes = param.Boolean(default=False, doc="""
        Whether to return Nodes or Graph.""")

    layout = param.Callable(default=None, doc="""
        A NetworkX layout function""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments passed to the layout function.""")

    def _process(self, element, key=None):
        if self.p.layout and isinstance(self.p.layout, FunctionType):
            import networkx as nx
            graph = nx.from_edgelist(element.array([0, 1]))
            positions = self.p.layout(graph, **self.p.kwargs)
            nodes = [tuple(pos)+(idx,) for idx, pos in sorted(positions.items())]
        else:
            source = element.dimension_values(0, expanded=False)
            target = element.dimension_values(1, expanded=False)
            nodes = np.unique(np.concatenate([source, target]))
            if self.p.layout:
                import pandas as pd
                df = pd.DataFrame({'index': nodes})
                nodes = self.p.layout(df, element.dframe(), **self.p.kwargs)
                nodes = nodes[['x', 'y', 'index']]
            else:
                nodes = circular_layout(nodes)
        if self.p.only_nodes:
            return Nodes(nodes)
        return element.clone((element.data, nodes))
        


class Graph(Dataset, Element2D):
    """
    Graph is high-level Element representing both nodes and edges.
    A Graph may be defined in an abstract form representing just
    the abstract edges between nodes and optionally may be made
    concrete by supplying a Nodes Element defining the concrete
    positions of each node. If the node positions are supplied
    the EdgePaths (defining the concrete edges) can be inferred
    automatically or supplied explicitly.

    The constructor accepts regular columnar data defining the edges
    or a tuple of the abstract edges and nodes, or a tuple of the
    abstract edges, nodes, and edgepaths.
    """

    group = param.String(default='Graph', constant=True)

    kdims = param.List(default=[Dimension('start'), Dimension('end')],
                       bounds=(2, 2))

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, tuple):
            data = data + (None,)* (3-len(data))
            edges, nodes, edgepaths = data
        else:
            edges, nodes, edgepaths = data, None, None
        if nodes is not None:
            node_info = None
            if isinstance(nodes, Nodes):
                pass
            elif not isinstance(nodes, Dataset) or nodes.ndims == 3:
                nodes = Nodes(nodes)
            else:
                node_info = nodes
                nodes = None
        else:
            node_info = None
        if edgepaths is not None and not isinstance(edgepaths, EdgePaths):
            edgepaths = EdgePaths(edgepaths)
        self._nodes = nodes
        self._edgepaths = edgepaths
        super(Graph, self).__init__(edges, kdims=kdims, vdims=vdims, **params)
        if self._nodes is None and node_info:
            nodes = self.nodes.clone(datatype=['pandas', 'dictionary'])
            if pd is None:
                for d in node_info.dimensions('value'):
                    nodes = nodes.add_dimension(d, len(nodes.vdims),
                                                node_info.dimension_values(d),
                                                vdim=True)
            else:
                node_info_df = node_info.dframe()
                node_df = nodes.dframe()
                idx = node_info.kdims[0].name
                node_df = pd.merge(node_df, node_info_df, left_on='index', right_on=idx)
                nodes = nodes.clone(node_df, vdims=node_info.vdims)
            self._nodes = nodes
        self._validate()
        self.redim = redim_graph(self, mode='dataset')


    def _validate(self):
        if self._edgepaths is None:
            return
        mismatch = []
        for kd1, kd2 in zip(self.nodes.kdims, self.edgepaths.kdims):
            if kd1 != kd2:
                mismatch.append('%s != %s' % (kd1, kd2))
        if mismatch:
            raise ValueError('Ensure that the first two key dimensions on '
                             'Nodes and EdgePaths match: %s' % ', '.join(mismatch))
        npaths = len(self._edgepaths.data)
        nedges = len(self)
        if nedges != npaths:
            mismatch = True
            if npaths == 1:
                edges = self.edgepaths.split()[0]
                vals = edges.dimension_values(0)
                npaths = len(np.where(np.isnan(vals))[0])
                if not np.isnan(vals[-1]):
                    npaths += 1
                mismatch = npaths != nedges
            if mismatch:
                raise ValueError('Ensure that the number of edges supplied '
                                 'to the Graph (%d) matches the number of '
                                 'edgepaths (%d)' % (nedges, npaths))


    def clone(self, data=None, shared_data=True, new_type=None, *args, **overrides):
        if data is None:
            data = (self.data, self.nodes)
            if self._edgepaths:
                data = data + (self.edgepaths,)
            overrides['plot_id'] = self._plot_id
        elif not isinstance(data, tuple):
            data = (data, self.nodes)
            if self._edgepaths:
                data = data + (self.edgepaths,)
        return super(Graph, self).clone(data, shared_data, new_type, *args, **overrides)


    def select(self, selection_specs=None, selection_mode='edges', **selection):
        """
        Allows selecting data by the slices, sets and scalar values
        along a particular dimension. The indices should be supplied as
        keywords mapping between the selected dimension and
        value. Additionally selection_specs (taking the form of a list
        of type.group.label strings, types or functions) may be
        supplied, which will ensure the selection is only applied if the
        specs match the selected object.

        Selecting by a node dimensions selects all edges and nodes that are
        connected to the selected nodes. To select only edges between the
        selected nodes set the selection_mode to 'nodes'.
        """
        selection = {dim: sel for dim, sel in selection.items()
                     if dim in self.dimensions('ranges')+['selection_mask']}
        if (selection_specs and not any(self.matches(sp) for sp in selection_specs)
            or not selection):
            return self

        index_dim = self.nodes.kdims[2].name
        dimensions = self.kdims+self.vdims
        node_selection = {index_dim: v for k, v in selection.items()
                          if k in self.kdims}
        nodes = self.nodes.select(**dict(selection, **node_selection))
        selection = {k: v for k, v in selection.items() if k in dimensions}

        # Compute mask for edges if nodes were selected on
        nodemask = None
        if len(nodes) != len(self.nodes):
            xdim, ydim = dimensions[:2]
            indices = list(nodes.dimension_values(2, False))
            if selection_mode == 'edges':
                mask1 = self.interface.select_mask(self, {xdim.name: indices})
                mask2 = self.interface.select_mask(self, {ydim.name: indices})
                nodemask = (mask1 | mask2)
                nodes = self.nodes
            else:
                nodemask = self.interface.select_mask(self, {xdim.name: indices,
                                                             ydim.name: indices})

        # Compute mask for edge selection
        mask = None
        if selection:
            mask = self.interface.select_mask(self, selection)

        # Combine masks
        if nodemask is not None:
            if mask is not None:
                mask &= nodemask
            else:
                mask = nodemask

        # Apply edge mask
        if mask is not None:
            data = self.interface.select(self, mask)
            if not np.all(mask):
                new_graph = self.clone((data, nodes))
                source = new_graph.dimension_values(0, expanded=False)
                target = new_graph.dimension_values(1, expanded=False)
                unique_nodes = np.unique(np.concatenate([source, target]))
                nodes = new_graph.nodes[:, :, list(unique_nodes)]
            paths = None
            if self._edgepaths:
                edgepaths = self._split_edgepaths
                paths = edgepaths.clone(edgepaths.interface.select_paths(edgepaths, mask))
                if len(self._edgepaths.data) == 1:
                    paths = paths.clone([paths.dframe() if pd else paths.array()])
        else:
            data = self.data
            paths = self._edgepaths
        return self.clone((data, nodes, paths))


    @property
    def _split_edgepaths(self):
        if len(self) == len(self._edgepaths.data):
            return self._edgepaths
        else:
            return self._edgepaths.clone(split_path(self._edgepaths))


    def range(self, dimension, data_range=True):
        if self.nodes and dimension in self.nodes.dimensions():
            node_range = self.nodes.range(dimension, data_range)
            if self._edgepaths:
                path_range = self._edgepaths.range(dimension, data_range)
                return max_range([node_range, path_range])
            return node_range
        return super(Graph, self).range(dimension, data_range)


    def dimensions(self, selection='all', label=False):
        dimensions = super(Graph, self).dimensions(selection, label)
        if selection == 'ranges':
            if self._nodes:
                node_dims = self.nodes.dimensions(selection, label)
            else:
                node_dims = Nodes.kdims+Nodes.vdims
                if label in ['name', True, 'short']:
                    node_dims = [d.name for d in node_dims]
                elif label in ['long', 'label']:
                    node_dims = [d.label for d in node_dims]
            return dimensions+node_dims
        return dimensions


    @property
    def nodes(self):
        """
        Computes the node positions the first time they are requested
        if no explicit node information was supplied.
        """
        if self._nodes is None:
            self._nodes = layout_nodes(self, only_nodes=True)
        return self._nodes


    @property
    def edgepaths(self):
        """
        Returns the fixed EdgePaths or computes direct connections
        between supplied nodes.
        """
        if self._edgepaths:
            return self._edgepaths
        if pd is None:
            paths = connect_edges(self)
        else:
            paths = connect_edges_pd(self)
        return EdgePaths(paths, kdims=self.nodes.kdims[:2])


    @classmethod
    def from_networkx(cls, G, layout_function, nodes=None, **kwargs):
        """
        Generate a HoloViews Graph from a networkx.Graph object and
        networkx layout function. Any keyword arguments will be passed
        to the layout function.
        """
        positions = layout_function(G, **kwargs)
        edges = G.edges()
        if nodes:
            idx_dim = nodes.kdims[-1].name
            xs, ys = zip(*[v for k, v in sorted(positions.items())])
            indices = list(nodes.dimension_values(idx_dim))
            edges = [(src, tgt) for (src, tgt) in edges if src in indices and tgt in indices]
            nodes = nodes.select(**{idx_dim: [eid for e in edges for eid in e]}).sort()
            nodes = nodes.add_dimension('x', 0, xs)
            nodes = nodes.add_dimension('y', 1, ys).clone(new_type=Nodes)
        else:
            nodes = Nodes([tuple(pos)+(idx,) for idx, pos in sorted(positions.items())])
        return cls((edges, nodes))


class Nodes(Points):
    """
    Nodes is a simple Element representing Graph nodes as a set of
    Points.  Unlike regular Points, Nodes must define a third key
    dimension corresponding to the node index.
    """

    kdims = param.List(default=[Dimension('x'), Dimension('y'),
                                Dimension('index')], bounds=(3, 3))

    group = param.String(default='Nodes', constant=True)


class EdgePaths(Path):
    """
    EdgePaths is a simple Element representing the paths of edges
    connecting nodes in a graph.
    """

    group = param.String(default='EdgePaths', constant=True)



class Chord(Graph):
    """
    Chord is a special type of Graph which lays out the nodes on a
    circle and connects the nodes using quadratic splines.
    """

    group = param.String(default='Chord')

    def __init__(self, data, kdims=None, vdims=None, **params):
        if isinstance(data, tuple):
            data = data + (None,)* (2-len(data))
            edges, nodes = data
        else:
            edges, nodes = data, None
        if nodes is not None:
            node_info = None
            if isinstance(nodes, Nodes):
                pass
            elif not isinstance(nodes, Dataset) or nodes.ndims == 3:
                nodes = Nodes(nodes)
            else:
                node_info = nodes
                nodes = None
        else:
            node_info = None
        super(Graph, self).__init__(edges, kdims=kdims, vdims=vdims, **params)
        nodes, edgepaths = compute_chords(self)
        self._nodes = Nodes(nodes)
        self._edgepaths = EdgePaths(edgepaths)
        if node_info:
            nodes = self.nodes.clone(datatype=['pandas', 'dictionary'])
            if pd is None:
                for d in node_info.dimensions('value'):
                    nodes = nodes.add_dimension(d, len(nodes.vdims),
                                                node_info.dimension_values(d),
                                                vdim=True)
            else:
                node_info_df = node_info.dframe()
                node_df = nodes.dframe()
                idx = node_info.kdims[0].name
                node_df = pd.merge(node_df, node_info_df, left_on='index', right_on=idx)
                nodes = nodes.clone(node_df, vdims=node_info.vdims)
            self._nodes = nodes
        self._validate()
        self.redim = redim_graph(self, mode='dataset')


    @property
    def edgepaths(self):
        return self._edgepaths


    @property
    def nodes(self):
        return self._nodes
