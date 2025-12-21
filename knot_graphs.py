from khovanov import *
import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def adjacency_graph(knot: Link, rad_increment=0.15):
    crossing_labels = [c.label for c in knot.crossings]
    G = nx.MultiDiGraph()
    G.add_nodes_from(crossing_labels)
    for c in knot.crossings:
        c0 = c.label
        adj = c.adjacent
        for i, (c1,s1) in zip(range(len(adj)), adj):
            # check if the arc has already been added
            if G.has_edge(c1.label, c0):
                data = G.get_edge_data(c1.label, c0)
                arc_already_exists = False
                for d in data.values():
                    if d['head_strand'] == i and d['tail_strand'] == s1:
                        arc_already_exists = True
                        break
                if arc_already_exists: continue
                    
            G.add_edge(c0, c1.label,
                           head_strand=s1, tail_strand=i)

    # update curvature for each edge
    for u,v in set(G.edges()):
        num_uv_edges = G.number_of_edges(u,v)
        if num_uv_edges == 1:
            G.edges[u,v,0]['rad'] = 0
        else:
            edge_keys = G.get_edge_data(u,v)
            possible_rads = [(i+1) * rad_increment for i in range(4)]
            for i,k in zip(range(len(edge_keys)), edge_keys):
                G.edges[u,v,k]['rad'] = (-1)**i * possible_rads[i//2]
        
    return G
    
def draw_adjacency_graph(knot: Link, head_pos=0.1, tail_pos=0.9, rad_increment=0.15):
    G = adjacency_graph(knot, rad_increment=rad_increment)
    #layout = nx.spring_layout(G)
    layout = nx.planar_layout(G)
    
    connectionstyle = [f'arc3,rad={attrs['rad']}'
                       for *edge, attrs in G.edges(keys=True, data=True)]
    rads = {(u,v,k) : attrs['rad']
            for u,v,k, attrs in G.edges(keys=True, data=True)}
                           
    head_strands = {(u,v,k) : attrs['head_strand'] 
                    for u,v,k, attrs in G.edges(keys=True, data=True)}
    tail_strands = {(u,v,k) : attrs['tail_strand'] 
                    for u,v,k, attrs in G.edges(keys=True, data=True)}

    nx.draw_networkx_nodes(G, pos=layout)
    nx.draw_networkx_labels(G, pos=layout)
    nx.draw_networkx_edges(G, pos=layout,
                           arrowsize=0.00001, connectionstyle=connectionstyle)

    my_draw_networkx_edge_labels(G, edge_labels=head_strands,
                                 pos=layout, label_pos=head_pos,
                                 rad=rads, rad_increment=rad_increment,
                                 bbox={"alpha": 0}, font_color='r', rotate=False)
    my_draw_networkx_edge_labels(G, edge_labels=tail_strands,
                                 pos=layout, label_pos=tail_pos,
                                 rad=rads, rad_increment=rad_increment,
                                 bbox={"alpha": 0}, font_color='r', rotate=False)

# Adapted from the following:
# Source - https://stackoverflow.com/a/70245742
# Posted by kcoskun, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-20, License - CC BY-SA 4.0
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0,
    rad_increment=0.15
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v, k): d for u, v, k, d in G.edges(keys=True, data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2, k), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])

        # get the rad for this particular edge
        my_rad = rad[n1,n2,k]
        if my_rad == 0:
            my_rad = rad_increment
        
        ctrl_1 = linear_mid + my_rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = label_pos*ctrl_mid_1 + (1.0-label_pos)*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

    