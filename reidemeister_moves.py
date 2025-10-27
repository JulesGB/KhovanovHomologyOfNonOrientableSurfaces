import sage.knots
import regina.engine

# Create a new link with a left/right R1 move on a given strand
def R1(link, strand_idx, sign=1):
    # convert to regina link
    rlink = regina.Link.fromPD(link.pd_code())
    strand = rlink.component(strand_idx)
    
    # do the move
    # r1(strand, side of strand{0,1}, sign{-1,1})
    if rlink.r1(strand, 0, sign):
        print('r1 performed')
    else:
        print('r1 unable to be performed')
    
    return sage.knots.link.Link(rlink.pdData())

def R2(link, edge1, edge2):

    # TODO - implement

    return

def R3(link, edge1, ):

    # TODO - implement

    return