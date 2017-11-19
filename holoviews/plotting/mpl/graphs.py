import param
import numpy as np

from matplotlib.collections import LineCollection

from ...core.util import basestring, unique_array
from ...core.options import Cycle
from ...element.graphs import search_indices
from ..util import rgb2hex
from .element import ColorbarPlot


class GraphPlot(ColorbarPlot):
    """
    GraphPlot
    """

    color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                  allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    edge_color_index = param.ClassSelector(default=None, class_=(basestring, int),
                                      allow_None=True, doc="""
      Index of the dimension from which the color will the drawn""")

    style_opts = ['edge_alpha', 'edge_color', 'edge_linestyle', 'edge_linewidth',
                  'node_alpha', 'node_color', 'node_edgecolors', 'node_facecolors',
                  'node_linewidth', 'node_marker', 'node_size', 'visible', 'cmap',
                  'edge_cmap']

    def _compute_styles(self, element, ranges, style):
        elstyle = self.lookup_options(element, 'style')
        color = elstyle.kwargs.get('node_color')
        cdim = element.nodes.get_dimension(self.color_index)
        if cdim:
            cs = element.nodes.dimension_values(self.color_index)
            # Check if numeric otherwise treat as categorical
            if cs.dtype.kind == 'f':
                style['c'] = cs
            else:
                factors = unique_array(cs)
                cs = search_indices(cs, factors)
                if isinstance(color, Cycle):
                    style['node_facecolors'] = [rgb2hex(color.values[v%len(color)])
                                                for v in cs]
                    style.pop('node_color', None)
                else:
                    style['c'] = cs
            if 'c' in style:
                self._norm_kwargs(element.nodes, ranges, style, cdim)
        elif color:
            style['c'] = style.pop('node_color')
        style['node_edgecolors'] = style.pop('node_edgecolors', 'none')

        edge_cdim = element.get_dimension(self.edge_color_index)
        if not edge_cdim:
            return style

        elstyle = self.lookup_options(element, 'style')
        cycle = elstyle.kwargs.get('edge_color')
        idx = element.get_dimension_index(edge_cdim)
        cvals = element.dimension_values(edge_cdim)
        if idx in [0, 1]:
            factors = element.nodes.dimension_values(2, expanded=False)
        else:
            factors = unique_array(cvals)
        if factors.dtype.kind == 'f':
            style['edge_array'] = cvals
        else:
            cvals = search_indices(cvals, factors)
            if isinstance(cycle, Cycle):
                factors = list(factors)
                style['edge_colors'] = [rgb2hex(cycle.values[v%len(cycle)])
                                        for v in cvals]
                style.pop('edge_color', None)
            else:
                style['edge_array'] = cvals
        if 'edge_array' in style:
            self._norm_kwargs(element, ranges, style, edge_cdim, 'edge_')
        if 'edge_vmin' in style:
            style['edge_clim'] = (style.pop('edge_vmin'), style.pop('edge_vmax'))
        return style

    def get_data(self, element, ranges, style):
        xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
        pxs, pys = (element.nodes.dimension_values(i) for i in range(2))
        dims = element.nodes.dimensions()
        self._compute_styles(element, ranges, style)

        paths = element.edgepaths.split(datatype='array', dimensions=element.edgepaths.kdims)
        if self.invert_axes:
            paths = [p[:, ::-1] for p in paths]
        return {'nodes': (pxs, pys), 'edges': paths}, style, {'dimensions': dims}

    def get_extents(self, element, ranges):
        """
        Extents are set to '' and None because x-axis is categorical and
        y-axis auto-ranges.
        """
        x0, x1 = element.nodes.range(0)
        y0, y1 = element.nodes.range(1)
        return (x0, y0, x1, y1)

    def init_artists(self, ax, plot_args, plot_kwargs):
        # Draw edges
        color_opts = ['c', 'cmap', 'vmin', 'vmax', 'norm']
        edge_opts = {k[5:] if 'edge_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if 'node_' not in k and k not in color_opts}
        paths = plot_args['edges']
        edges = LineCollection(paths, **edge_opts)
        ax.add_collection(edges)

        # Draw nodes
        xs, ys = plot_args['nodes']
        node_opts = {k[5:] if 'node_' in k else k: v
                     for k, v in plot_kwargs.items()
                     if 'edge_' not in k}
        if 'size' in node_opts: node_opts['s'] = node_opts.pop('size')**2
        nodes = ax.scatter(xs, ys, **node_opts)

        return {'nodes': nodes, 'edges': edges}

    def update_handles(self, key, axis, element, ranges, style):
        nodes = self.handles['nodes']
        data, style, axis_kwargs = self.get_data(element, ranges, style)
        xs, ys = data['nodes']
        nodes.set_offsets(np.column_stack([xs, ys]))
        cdim = element.nodes.get_dimension(self.color_index)
        if cdim:
            nodes.set_clim((style['vmin'], style['vmax']))
            nodes.set_array(style['c'])
            if 'norm' in style:
                nodes.norm = style['norm']

        edges = self.handles['edges']
        paths = data['edges']
        edges.set_paths(paths)
        edges.set_visible(style.get('visible', True))
        cdim = element.get_dimension(self.edge_color_index)
        if cdim:
            edges.set_clim((style['edge_vmin'], style['edge_vmax']))
            edges.set_array(style['edge_c'])
            if 'norm' in style:
                edges.norm = style['edge_norm']


        return axis_kwargs
