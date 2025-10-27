import sage.knots
import regina.engine

def R1(link, arc_idx, strand_idx=0, sign=1):
    """
    Create a new link with a left/right R1 move performed on a given arc.
    Args:
        link : a link
        arc_idx : idx indicating the arc of the given component
        strand_idx : idx indicating component of link, default 0
        sign : sign of new crossing (either -1 or 1), default 1
    Returns:
        A sage link
    """
    
    # convert to regina link
    rlink = regina.Link.fromPD(link.pd_code())

    # locate the relevant arc in the component
    arc = rlink.component(strand_idx)
    for _ in range(arc_idx):
        arc = arc.next()
    
    # do the move
    # r1(arc, side of strand{0,1}, sign{-1,1})
    if rlink.r1(arc, 0, sign):
        print('r1 performed')
    else:
        print('r1 unable to be performed')

    #convert back to sage link
    return sage.knots.link.Link(rlink.pdData())

def R2(link, arc1, arc2):

    # TODO - implement

    return

def R3(link):

    # TODO - implement

    return