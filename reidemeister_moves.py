from sage.knots.link import *
import regina.engine.Link

def R1(link, edge):
    pd = link.pd_code().copy() # don't change input knot

    # find total number of edges, i.e., the maximum edge label in pd
    max_edge_label = max([item for row in pd for item in row])

    # new edges alpha<beta
    alpha, beta = max_edge_label+1, max_edge_label+2

    # [+i, -beta, -alpha, +alpha]
    pd.append([edge,beta,alpha,alpha])

    # TODO - implement
    print(pd)
    return Link(pd)

def R1_regina(link, edge):
    pd = link.pd_code().copy()
    link_regina = regina.Link.fromPD(pd)
    link_regina.withR1(edge, 

def R2(link, edge1, edge2):
    pd = link.pd_code().copy()

    # TODO - implement

    return

def R3(link, edge1, ):
    pd = link.pd_code().copy()

    # TODO - implement

    return