#   Khovanov Cobordism Calculator
#   =============================
#   A Python module to calculate cobordism maps induced on Khovanov homology.
#   - Written by Zsombor Fehér, 2024.
#
#
#   Quick Start Guide
#   =================
#   Example: Distinguishing two ribbon disks of 6_1, given as two band diagrams on the same link diagram.
#
#   1. Make sure you have SciPy and NumPy installed and updated to the latest version:
#           pip install --upgrade scipy
#      (Some methods require Sage to be installed, but tasks like this example do not.)
#   2. Start a Python session, copy this file to your working folder, and import this module:
#           from khovanov import *
#   3. Obtain a PD code of the link, and determine how the PD code corresponds to the diagram. 
#      Label each crossing in your diagram with its corresponding index in the PD code.
#      Mark the 0th strand of each crossing with a dot, corresponding to the 0th element of its PD code entry.
#      (Since this module does not have a graphical interface yet, you might find it useful to draw the link in
#      SnapPy's Plink Editor, and enable Info -> DT labels and Info -> PD code for getting this correspondence.)
#   4. Create a Link object using the PD code:
#           L = Link([(9, 4, 10, 5), (5, 8, 6, 9), (11, 2, 12, 3), (3, 10, 4, 11), (1, 7, 2, 6), (7, 1, 8, 12)])
#      You can verify that your diagram is marked correctly using print(L).
#   5. Create the two Cobordism objects based on the band diagrams and your marking of crossings.
#      For example, S0.band_move(-1, (0, 0), (2, 1)) represents a (-1)-twisted band, connecting the 0th crossing's
#      0th strand to the 2nd crossing's 1st strand, positioned on the band's left side.
#           S0 = Cobordism(L)
#           S0.band_move(-1, (0, 0), (2, 1))
#           S0.finish()
#           S1 = Cobordism(L)
#           S1.band_move(-1, (1, 2), (3, 3))
#           S1.finish()
#      You can check the resulting movie of the cobordism with print(S0).
#   6. Calculate the Khovanov-Jacobsson classes (CKhElement objects) and compare them in homology:
#           compare(S0.KJ_class(), S1.KJ_class())
#   7. If this fails to distinguish the surfaces, try mirroring the cobordisms:
#           compare(S0.mirror().KJ_class(), S1.mirror().KJ_class())
#
#
#   Hierarchy of Classes
#   ====================
#
#       Crossing           smoothing == 0        smoothing == 1
#          2                     2                     2
#          |                     |                     |
#     3 ------- 1           3 --/ /-- 1           3 --\ \-- 1
#          |                     |                     |
#          0                     0                     0
#
#        Strand
#     0 ---o--- 1
#
#    CrossingStrand (CS) --crossing------> Crossing | Strand
#       ---o             --strand_index--> int
#
#                    crossings
#              Link -----------> list[Crossing, Strand]                         L
#                ^                                                              /\
#                |                                                             /  \
#           link |                                                            /    \
#                |      --smoothing--> dict[Crossing, int]                   /      \
#            SmoothLink --loops------> list[tuple[CS]]                     SL        SL
#                ^      --loop_of----> dict[CS, tuple[CS]]                  |        /|\
#                |                                                          |       / | \
#    smooth_link |                                                          |      /  |  \
#                |                                                          |     /   |   \
#        LabelledSmoothing --labels-------> dict[tuple[CS], str]           LS   LS   LS   LS
#                ^         --coefficient--> int                              \   |   /   /
#                |                                                            \  |  /  /
#            [i] |                                                             \ | / /
#                |                                                              \|//
#            CKhElement                                                         CKH
#
#    Cobordism ----links----> list[Link]
#              ----movie----> list[tuple[Any]]
#
#
#   Methods of CKhElement                                                          Methods of Link
#   =====================                                                          ===============
#
#   morse_birth() -> cs                                                            -> loop
#
#                                                    -----
#                                     -->           |     |
#                                                    --o-- cs
#
#   morse_death(cs)
#
#             -----
#            |     |                  -->
#             --o-- cs
#
#   morse_saddle(cs0, cs1) | (cs1, cs0)                                            -> cs2
#
#           cs0 ----------                        ---\   /--- cs2
#                                     -->             | |
#           cs1 ----------                        ---/   \---
#
#   fuse(s)
#
#               s
#           ----o----                 -->          ---------
#
#   reidemeister_1(cs0 | cs1) -> s | None                                          -> positive_crossing, (s | None)
#
#           cs0 ---
#              |   |
#           ---+--- cs1               -->           ----
#              |                                        |
#              |                                        |
#
#   reidemeister_1_up(cs, True | False) -> cs0                                     -> cs0
#
#                                               cs0 3 ---            cs0 2 ---
#           cs                                       |   |                |   |
#            ----                     -->       0 ---|--- 2    or    3 ------- 1
#                |                                   |                    |
#                |                                 1 |                  0 |
#
#   reidemeister_2(cs0, cs1) | (cs1, cs0) -> cs2, cs3                              -> c, d, cs2, cs3
#
#               cs0
#       ----\ /--------\ /----                      ---------------- cs3
#          c X        d X             -->
#       ----/ \--------/ \----                  cs2 ----------------
#               cs1
#
#   reidemeister_2_up(cs0, cs1) -> cs2, cs3                                        -> c, d
#
#                                                   3  2 cs3   0   3
#       cs0 ----------------                     ----\ /--------\ /----
#                                     -->           c \        d /
#       cs1 ----------------                     ----/ \--------/ \----
#                                                   0  1 cs2   1   2
#
#   reidemeister_3(cs0, cs1, cs2)                                                  -> positive_orientation
#
#         \           /                                   \   /
#          \  cs0    /                                     \ /
#       -----------------                                   X
#         cs1\     /cs2                                    / \
#             \   /                   -->                 /   \
#              \ /                                       /     \
#               X                                   -----------------
#              / \                                     /         \
#             /   \                                   /           \
#
#
#   Methods of Cobordism
#   ====================
#   In addition to the above methods of CKhElement:
#
#   band_move(n, cs0, (cs1, True), (cs2, False), ..., csk)
#
#        cs0  cs1  cs2     csk                   cs0  cs1  cs2                csk
#         |    |    |       |                     |    |    |          n=3     |
#         |    |    |       |                     \---------|--   --\ /\ /\ /--/
#         |    |    |  ...  |         -->              |    |  ...   \  \  \
#         |    |    |       |                     /---------|--   --/ \/ \/ \--\
#         |    |    |       |                     |    |    |                  |
#
#   finish()
#
#             -----
#            |   --|--\ /--\
#            |  |  |   \    |         -->
#             -----   / \--/
#                \---/
#
#   reverse()
#       returns the reverse of the Cobordism (i.e. turns it upside down)
#   mirror()
#       returns the mirror of the Cobordism
#   map(CKH)
#       applies the cobordism map to a CKhElement
#   KJ_class()
#       returns the Khovanov-Jacobsson class
#
#
#   Acknowledgments
#   ===============
#   Link class based on SnapPy's implementation by Nathan Dunfield.
#   Additional ideas by Alan Du and Gary Dunkerley.


import copy
from collections import OrderedDict, namedtuple
from itertools import combinations

try:
    from sage.all import matrix, ChainComplex, ZZ   # type: ignore
    _within_sage = True
except:
    _within_sage = False
# decorator for functions that need Sage
def needs_sage(function):
    def wrapped_function(*args, **kwargs):
        if _within_sage:
            return function(*args, **kwargs)
        else:
            raise Exception("Sorry, this function needs Sage.")
    return wrapped_function


class Crossing:
    """
    A crossing of the link, connected to 4 other crossings. Its properties:
    * label: Arbitrary name used for identifying and printing the crossing.
    * adjacent: The list of tuples (Crossing | Strand, int) that this Crossing is attached to.
    """

    __slots__ = ("label", "adjacent")

    def __init__(self, label=None):
        self.label = label
        self.adjacent = [None, None, None, None]

    def __getitem__(self, i):
        return (self, i % 4)

    def __setitem__(self, i, other):
        o, j = other
        self.adjacent[i % 4] = other
        o.adjacent[j] = (self, i % 4)

    def __repr__(self):
        return str(self.label)

    def print_long(self):
        """
        Prints the adjacency of self.
        """
        print("<%s : %s>" % (self.label, self.adjacent))

    def crossing_strands(self):
        """
        Returns the list of 4 CrossingStrands around self.
        """
        return [CrossingStrand(self, v) for v in range(4)]

    def flip(self):
        """
        Rotates the crossing 180°.
        """
        self[2], self[3], self[0], self[1] = self.adjacent


class Strand:
    """
    Similar to Crossing, but only 2 ends.
    """

    __slots__ = ("label", "adjacent")

    def __init__(self, label=None):
        self.label = label
        self.adjacent = [None, None]

    def __getitem__(self, i):
        return (self, i % 2)

    def __setitem__(self, i, other):
        o, j = other
        self.adjacent[i % 2] = other
        o.adjacent[j] = (self, i)

    def __repr__(self):
        return str(self.label)

    def print_long(self):
        """
        Prints the adjacency of self.
        """
        print("<%s : %s>" % (self.label, self.adjacent))

    def crossing_strands(self):
        """
        Returns the list of 2 CrossingStrands around self.
        """
        return [CrossingStrand(self, v) for v in range(2)]

    def fuse(self):
        """
        Joins the incoming and outgoing strands to remove self from the picture.
        """
        (a, i), (b, j) = self.adjacent
        a.adjacent[i] = (b, j)
        b.adjacent[j] = (a, i)

    def is_loop(self):
        """
        Returns whether self is connected to itself.
        """
        return self == self.adjacent[0][0]


class CrossingStrand(namedtuple("CS", ("crossing", "strand_index"))):
    """
    One of the four/two strands of a Crossing/Strand.
    """

    def opposite(self):
        """
        The CrossingStrand at the other end of the edge from self.
        """
        return CrossingStrand(*self.crossing.adjacent[self.strand_index])

    def next_in_smoothing(self, smoothing: int):
        """
        The CrossingStrand next to self in the same crossing according to the smoothing.
        """
        c, i = self.crossing, self.strand_index
        if smoothing == 0:
            j = [1, 0, 3, 2][i]
        elif smoothing == 1:
            j = [3, 2, 1, 0][i]
        else:
            raise ValueError("Smoothing needs to be 0 or 1.")
        return CrossingStrand(c, j)

    def __repr__(self):
        return "%s-%s" % (self.crossing, self.strand_index)

    def next(self):
        """
        The CrossingStrand opposite from self in the same Crossing / Strand.
        """
        if isinstance(self.crossing, Strand):
            return CrossingStrand(self.crossing, (self.strand_index + 1) % 2)
        elif isinstance(self.crossing, Crossing):
            return CrossingStrand(self.crossing, (self.strand_index + 2) % 4)

    def __add__(self, i: int):
        """
        Add an integer to the strand_index, crossing remains the same.
        """
        if isinstance(self.crossing, Strand):
            return CrossingStrand(self.crossing, (self.strand_index + i) % 2)
        elif isinstance(self.crossing, Crossing):
            return CrossingStrand(self.crossing, (self.strand_index + i) % 4)

    def __sub__(self, i: int):
        """
        Subtract an integer from the strand_index, crossing remains the same.
        """
        if isinstance(self.crossing, Strand):
            return CrossingStrand(self.crossing, (self.strand_index - i) % 2)
        elif isinstance(self.crossing, Crossing):
            return CrossingStrand(self.crossing, (self.strand_index - i) % 4)


class Link:
    """
    Links are made from Crossings and Strands. The general model is that of the PD
    diagrams used in KnotTheory: http://katlas.org/wiki/Planar_Diagrams.
    Here are two ways of creating the figure-8 knot, first via a PD code:
    >>> K1 = Link([[8, 3, 1, 4], [2, 6, 3, 5], [6, 2, 7, 1], [4, 7, 5, 8]])

    and by directly gluing up Crossings:
    >>> a, b, c, d = [Crossing(x) for x in 'abcd']
    >>> a[0], a[1], a[2], a[3] = c[1], d[0], b[1], b[0]
    >>> b[2], b[3] = d[3], c[2]
    >>> c[3], c[0] = d[2], d[1]
    >>> K2 = Link([a, b, c, d])

    You can also construct a link by taking the closure of a braid:
    >>> Link(braid_closure=[1, 2, -1, -2])

    The main differences from SnapPy's Link class are as follows:
    - Unlinked unknot components are not automatically discarded.
    - Strands are allowed and are not necessarily fused.
    - Many unnecessary properties and methods have been removed.
    """

    def __init__(self, crossings=None, braid_closure=None, fuse=True):
        if crossings is not None and braid_closure is not None:
            raise ValueError("Specified *both* crossings and braid_closure.")

        if crossings is not None:
            if isinstance(crossings, str):
                raise Exception("Removed functionality of creating a Link from string.")
            # Crossings can be a PD code rather than a list of actual crossings
            if len(crossings) > 0 and not isinstance(crossings[0], (Strand, Crossing)):
                crossings = self._crossings_from_PD_code(crossings)
        elif braid_closure is not None:
            crossings = self._crossings_from_braid_closure(braid_closure)
        else:
            crossings = []  # empty link

        # Make sure everything is tied up.
        if True in [None in c.adjacent for c in crossings]:
            raise ValueError("No loose strands allowed")

        # Fuse the strands. If there is a component made up only of one Strand, we keep it.
        self.crossings = []
        for s in crossings:
            if fuse and isinstance(s, Strand) and not s.is_loop():
                s.fuse()
                continue
            self.crossings.append(s)

        # If none of the crossings are labelled, then label them.
        if all(X.label is None for X in self.crossings):
            for c, X in enumerate(self.crossings):
                X.label = c

    def _crossings_from_PD_code(self, code):
        gluings = OrderedDict()

        for c, X in enumerate(code):
            for i, x in enumerate(X):
                if x in gluings:
                    gluings[x].append((c, i))
                else:
                    gluings[x] = [(c, i)]

        if {len(v) for v in gluings.values()} != {2}:
            raise ValueError("PD code isn't consistent")

        crossings = [Crossing(i) for i in range(len(code))]
        for (c, i), (d, j) in gluings.values():
            crossings[c][i] = crossings[d][j]

        return crossings

    def _crossings_from_braid_closure(self, word, num_strands=None):
        """
        Compute the braid closure of a word given in the form of a list of
        integers, where 1, 2, 3, etc correspond to the generators
        sigma_1, sigma_2, sigma_3, and so on, and negative numbers to
        their inverses.
        """
        #       ----------\ /--
        # sigma_2    0     \ 
        #       -----\ /--/ \--
        # sigma_1     /   0
        #       -----/ \-------

        if num_strands is None:
            num_strands = max([abs(a) for a in word]) + 1

        strands = [Strand(i) for i in range(num_strands)]
        current = [(x, 1) for x in strands]
        crossings = []

        for i, a in enumerate(word):
            C = Crossing(i)
            crossings.append(C)
            if a < 0:
                t0, t1 = 1, 0
                b0, b1 = 2, 3
            else:
                t0, t1 = 0, 3
                b0, b1 = 1, 2
            j0, j1 = abs(a) - 1, abs(a)
            C[t0] = current[j0]
            C[t1] = current[j1]
            current[j0] = C[b0]
            current[j1] = C[b1]

        for i in range(num_strands):
            strands[i][0] = current[i]

        for s in strands:
            s.fuse()
        return crossings

    def crossing_strands(self, make_set=False, ignore=set()) -> list[CrossingStrand] | set[CrossingStrand]:
        """
        Returns a list (or set if make_set is set to True) of all CrossingStrands (excluding elements of ignore).
        """
        # We don't make this into a set by default, because then loops would appear in a different random order every run.
        def add(cs):
            if cs not in ignore:
                result.add(cs)

        def append(cs):
            if cs not in ignore:
                result.append(cs)

        if make_set:
            result = set()
            for c in self.crossings:
                if isinstance(c, Crossing):
                    for i in range(4):
                        add(CrossingStrand(c, i))
                else:
                    for i in range(2):
                        add(CrossingStrand(c, i))
        else:
            result = []
            for c in self.crossings:
                if isinstance(c, Crossing):
                    for i in range(4):
                        append(CrossingStrand(c, i))
                else:
                    for i in range(2):
                        append(CrossingStrand(c, i))
        return result

    def __repr__(self):
        """
        Returns the adjacency table of the link.
        """
        n = len(self.crossings)
        result = f"Link of {n} crossing{'s' if n != 1 else ''}{':' if n>0 else '.'}"
        for c in self.crossings:
            result += f"\n{c} {[(a, i) if a in self.crossings else None for (a, i) in c.adjacent]}"
        return result

    # def view(self):
    #     import snappy
    #     snap = {c: snappy.Crossing(c.label) for c in self.crossings}
    #     for c in self.crossings:
    #         for i, cs in enumerate(c.adjacent):
    #             snap[c][i] = cs
    #     L = snappy.Link(list(snap.values()))
    #     L.view()      # this doesn't open the plink window for some reason

    def PD_code(self):
        """
        Returns a PD code of the link.
        Note that currently it does not use the conventions for orientations used by SnapPy.
        """
        remaining = self.crossing_strands()
        strand_labels = {}
        label = 0
        while len(remaining) > 0:
            start = remaining[0].opposite()
            cs = start
            while True:
                strand_labels[cs] = label
                remaining.remove(cs)
                cs = cs.opposite()

                strand_labels[cs] = label
                remaining.remove(cs)
                cs = cs.next()
                label += 1

                if cs == start:
                    break
        return [[strand_labels[cs] for cs in c.crossing_strands()] for c in self.crossings]

    def copy(self, recursively=False):
        """
        Returns a copy of the link.
        """
        if recursively:
            return copy.deepcopy(self)
        else:
            crossings = [type(c)(c.label) for c in self.crossings]
            loose_strands = set()
            for n in range(len(crossings)):
                strand_indices = [0, 1, 2, 3] if isinstance(crossings[n], Crossing) else [0, 1]
                for i in strand_indices:
                    loose_strands.add((n, i))
            while loose_strands:
                n, i = loose_strands.pop()
                adj_c, adj_i = self.crossings[n].adjacent[i]
                adj_n = self.crossings.index(adj_c)
                crossings[n][i] = crossings[adj_n][adj_i]
                loose_strands.remove((adj_n, adj_i))

            link = Link(crossings=crossings, fuse=False)
            return link

    def mirror(self, use_stored=False):
        """
        Returns the mirror image of the link by rotating every crossing by -90°.
        """
        if use_stored and hasattr(self, "mirror_link"):
            return self.mirror_link

        new_crossings = dict()
        for C in self.crossings:
            C_new = type(C)(label=C.label)
            new_crossings[C] = C_new

        def convert(C, c):
            # Go from a crossing to the mirror.
            if isinstance(C, Crossing):
                return new_crossings[C], (c + 1) % 4
            else:
                return new_crossings[C], c

        for A in self.crossings:
            strand_indices = [0, 1, 2, 3] if isinstance(A, Crossing) else [0, 1]
            for a in strand_indices:
                B, b = A.adjacent[a]
                B_new, b_new = convert(B, b)
                B_new[b_new] = convert(A, a)

        new_link = type(self)(list(new_crossings.values()))
        if hasattr(self, "n_plus"):
            new_link.n_plus = self.n_minus
            new_link.n_minus = self.n_plus
        self.mirror_link = new_link
        # new_link.mirror_link = self       # might cause trouble for rotating 90° instead of -90°
        return new_link

    def LS_data_to_class(self, LS_data: tuple):
        """
        Returns a LabelledSmoothing object constructed from data obtained by to_data().
        """
        assert all(isinstance(c, Crossing) for c in self.crossings)   # Strands not handled correctly yet
        n = len(self.crossings)
        SL = SmoothLink(self, LS_data[:n])
        LS = LabelledSmoothing(SL, ["1"] * len(SL.loops))
        crossing_strands = [CrossingStrand(c, j) for c in self.crossings for j in (0, 2)]
        for i, cs in enumerate(crossing_strands):
            loop = SL.loop_of[cs]
            LS.labels[loop] = "1" if LS_data[n + i] == "1" else "x"
        return LS

    def find_cs(self, cs):
        """
        Returns the corresponding CrossingStrand from another Link instance,
        based on the label of the crossing.
        * cs: a CrossingStrand/tuple (crossing, index), where
              crossing is a Crossing/Strand object, or a label.
        """
        # Hence it is important that all labels are different in each link of a Cobordism.
        label = cs[0].label if isinstance(cs[0], Crossing | Strand) else cs[0]
        for new_crossing in self.crossings:
            if new_crossing.label == label:
                return CrossingStrand(new_crossing, cs.strand_index)
        raise ValueError("Label not found.")

    def find_crossing(self, crossing):
        """
        Returns the corresponding Crossing/Strand from another Link instance,
        based on the label of the crossing.
        * crossing: a Crossing/Strand object, or a label.
        """
        label = crossing.label if isinstance(crossing, Crossing | Strand) else crossing
        for new_crossing in self.crossings:
            if new_crossing.label == label:
                return new_crossing
        raise ValueError("Label not found.")

    def morse_birth(self):
        """
        Adds a new unlinked unknot component, consisting of one Strand.
        Returns the new loop.
        """
        n = len(self.crossings)
        labels = {c.label for c in self.crossings}
        while n in labels:
            n += 1
        s = Strand(n)
        s[0] = s[1]
        self.crossings.append(s)
        return (CrossingStrand(s, 0), CrossingStrand(s, 1))

    def morse_death(self, cs: CrossingStrand):
        """
        Removes the loop of a CrossingStrand if it consists of one Strand.
        """
        s = cs.crossing
        if not s.is_loop():
            raise ValueError("Can only do Morse death at a component of only one Strand.")
        self.crossings.remove(s)

    def morse_saddle(self, cs0: CrossingStrand, cs1: CrossingStrand):
        """
        Does a Morse saddle move between the two strands specified by the CrossingStrands.
        Adds a new Strand if needed. Returns a CrossingStrand on the opposite loop.
        We don't check these, but the CrossingStrands have to:
         - be in the boundary of the same region,
         - correspond to adjacent strands in the saddle (see picture).
        """
        # cs0 o---------        c0 o--\   /-- c3
        #                  ->          | |
        # cs1 o---------        c1 o--/   \-- c2

        c0, i0 = cs0
        c1, i1 = cs1
        c2, i2 = cs1.opposite()
        c3, i3 = cs0.opposite()
        if c0 != c2:
            c0[i0] = c1[i1]
            c2[i2] = c3[i3]
            return CrossingStrand(c3, i3)
        else:
            n = len(self.crossings)
            labels = {c.label for c in self.crossings}
            while n in labels:
                n += 1
            s = Strand(n)
            s[0] = s[1]
            self.crossings.append(s)
            return CrossingStrand(s, 0)

    def fuse(self, strand: Strand):
        """
        Updates self.crossings according to a Strand.fuse().
        """
        # TODO maybe change all fuse functions' Strand parameters to CrossingStrand to make it more uniform?
        assert not strand.is_loop()
        strand.fuse()
        self.crossings.remove(strand)

    def reidemeister_1(self, cs: CrossingStrand):
        """
        Untwists a Reidemeister-1 loop consisting of only Strands and one Crossing.
        Adds a new Strand if there are no other crossings on the loop.
        Returns the sign of the crossing (True if it was positive), and the new Strand (or None if it was fused).
        """
        #      i ---
        #       |   |
        # i1 -------          ->         ----
        #       |                            |
        #       i0                           |

        # check that loop has only Strands and get the sign of the Crossing
        # TODO probably can simplify, assuming that the loop has no Strands?
        crossing, index = cs
        assert isinstance(crossing, Crossing)
        next_cs = cs.opposite()
        remove = [crossing]
        while isinstance(next_cs.crossing, Strand):
            remove.append(next_cs.crossing)
            next_cs = next_cs.next().opposite()
        other_crossing, other_index = next_cs
        assert other_crossing == crossing
        if {index, other_index} in ({0, 1}, {2, 3}):
            positive_crossing = True
        else:
            assert {index, other_index} in ({1, 2}, {3, 0})
            positive_crossing = False

        # create new Strand and connect it to the Crossings
        # TODO don't create a Strand by default
        s = Strand(str(crossing.label) + "s")
        assert s.label not in {c.label for c in self.crossings}
        i = index if index % 2 == 0 else other_index
        i0 = (i + 2) % 4
        i1 = (i - 1) % 4 if positive_crossing else (i + 1) % 4
        cs0 = CrossingStrand(crossing, i0)
        cs1 = CrossingStrand(crossing, i1)
        opp0 = cs0.opposite()
        if opp0 != cs1:
            s[0] = tuple(opp0)
            s[1] = tuple(cs1.opposite())
        else:
            s[0] = s[1]
        for c in remove:
            self.crossings.remove(c)
        self.crossings.append(s)
        if not s.is_loop():
            self.fuse(s)
            return positive_crossing, None
        return positive_crossing, s

    def reidemeister_1_up(self, cs: CrossingStrand, positive_crossing: bool):
        """
        Adds a new Reidemeister-1 loop to the left of the given CrossingStrand of the given sign.
        Returns the CrossingStrand on the left of the new loop.
        """
        #                           cs0 ---
        #   cs                         |   |
        #    ----         ->        ---|---
        #        |                     |
        #        |                     |

        # get unique label for the new crossing
        n = len(self.crossings)
        labels = {c.label for c in self.crossings}
        while n in labels:
            n += 1
        new = Crossing(n)
        self.crossings.append(new)

        # s = cs.crossing
        # fuse = isinstance(s, Strand) and s.is_loop()

        # update crossing adjacency
        if positive_crossing:
            new[1] = tuple(cs.opposite())
            new[2] = new[3]
            new[0] = tuple(cs)
            cs0 = CrossingStrand(new, 3)
        else:
            new[0] = tuple(cs.opposite())
            new[1] = new[2]
            new[3] = tuple(cs)
            cs0 = CrossingStrand(new, 2)
        # if fuse:
        #     self.fuse(s)
        return cs0

    def reidemeister_2(self, cs0: CrossingStrand, cs1: CrossingStrand):
        """
        Does a Reidemeister-2 simplification between two strands specified by the CrossingStrands.
        Adds a new Strand if there are no other crossings on a strand.
        Returns c, d, cs2, cs3.
        The CrossingStrands have to:
        - be two adjacent strands of the same crossing,
        - their opposite() must be on the same crossing, with the correct strand over.
        """
        #            cs0
        #    k3   i3 i0   j0 j3   l3
        #  c3 -----\ /-----\ /----- d3                ----------- cs3
        #         c \       / d            ->
        #  c2 -----/ \-----/ \----- d2            cs2 -----------
        #    k2   i2 i1   j1 j2   l2
        #            cs1

        # get relevant crossings and indices
        if cs1 == cs0 + 1:
            cs0, cs1 = cs1, cs0
        c,  i0 = cs0
        c1, i1 = cs1
        assert c == c1 and (i0 + i1) % 2 == 1  # first condition above
        d,  j0 = cs0.opposite()
        d1, j1 = cs1.opposite()
        assert d == d1 and (i0 + j0) % 2 == 0  # second condition above
        i2, i3 = (i0 + 2) % 4, (i1 + 2) % 4
        j2, j3 = (j0 + 2) % 4, (j1 + 2) % 4
        c2, k2 = c.adjacent[i2]
        c3, k3 = c.adjacent[i3]
        d2, l2 = d.adjacent[j2]
        d3, l3 = d.adjacent[j3]

        # update self.crossings and crossing adjacencies
        self.crossings.remove(c)
        self.crossings.remove(d)
        if c2 not in (c, d) and d3 not in (c, d):
            # general case, don't need to add any Strands
            c2[k2] = d2[l2]
            c3[k3] = d3[l3]
            cs2 = CrossingStrand(c2, k2)
            cs3 = CrossingStrand(d3, l3)
        elif c2 == c and d3 == d:
            # everything is one loop, we add a Strand in the middle
            s = Strand(str(c.label) + "s")
            assert s.label not in {c.label for c in self.crossings}
            s[0] = s[1]
            self.crossings.append(s)
            cs2 = cs3 = CrossingStrand(s, 0)
        elif c2 == c:
            # general case variant
            d2[l2] = d3[l3]
            cs2 = cs3 = CrossingStrand(d2, l2)
        elif d3 == d:
            # general case variant
            c2[k2] = c3[k3]
            cs2 = cs3 = CrossingStrand(c2, k2)
        else:
            if c2 != d:
                # general case variant
                c2[k2] = d2[l2]
                cs2 = CrossingStrand(c2, k2)
            else:
                # bottom is a loop, we add a Strand there
                s = Strand(str(c.label) + "s")
                assert s.label not in {c.label for c in self.crossings}
                s[0] = s[1]
                self.crossings.append(s)
                cs2 = CrossingStrand(s, 0)
            if d3 != c:
                # general case variant
                c3[k3] = d3[l3]
                cs3 = CrossingStrand(d3, l3)
            else:
                # top is a loop, we add a Strand there
                t = Strand(str(d.label) + "s")
                assert t.label not in {c.label for c in self.crossings}
                t[0] = t[1]
                self.crossings.append(t)
                cs3 = CrossingStrand(t, 0)

        return c, d, cs2, cs3

    def reidemeister_2_up(self, cs0: CrossingStrand, cs1: CrossingStrand):
        """
        Adds two new Crossings by a Reidemeister-2 move, moving the CrossingStrand cs0 right and over cs1.
        Returns the new Crossings c and d.
        """
        #   cs0 -----------                -----\ /-----\ /-----
        #                        ->            c \     d /
        #   cs1 -----------                -----/ \-----/ \-----

        # get unique labels for the two new crossings
        n = len(self.crossings)
        labels = {c.label for c in self.crossings}
        while n in labels:
            n += 1
        m = n + 1
        while m in labels:
            m += 1
        c = Crossing(n)
        d = Crossing(m)
        self.crossings.append(c)
        self.crossings.append(d)

        # update crossing adjacency
        cs0opp = cs0.opposite()
        if cs0opp != cs1:
            d[2], d[3] = tuple(cs1.opposite()), tuple(cs0opp)
        else:
            d[2] = d[3]
        c[0], c[1], c[2], c[3] = tuple(cs1), d[1], d[0], tuple(cs0)
        # return CrossingStrand(c, 1), CrossingStrand(c, 2)     , CrossingStrand(d, 3)
        return c, d

    def reidemeister_3(self, cs0: CrossingStrand, cs1: CrossingStrand, cs2: CrossingStrand):
        """
        Does a Reidemeister-3 move on a small triangle specified by the three CrossingStrands.
        Returns True if the triangle's strands are laid down positively (figure below and Table 5(a)).
        The CrossingStrands have to be as in the figure, i.e.
        - form a small triangle, with cs0, cs1, cs2 being a positive orientation,
        - have two overstrands on the cs0 side of the triangle,
        - cs1.crossing == cs0.crossing and cs2.crossing be on the cs0 strand.
        """
        #      C1            D0                     C1   D0
        #       \   cs0     /                        \ e /   
        #      c \  i0 j1  / d                        \ /    
        #  C0 ----------------- D1                     /        
        #        i1\     /j0                        k1/ \k0       
        #     cs1   \   /   cs2        ->            /   \ 
        #          k0\ /k1                        j0/     \i1
        #             /                     C0 ----------------- D1 
        #            / \                        d /  j1 i0  \ c
        #           / e \                        /           \ 
        #          E1   E0                      E1           E0

        # get crossings of triangle
        c,  i0 = cs0
        c0, i1 = cs1
        d,  j0 = cs2
        d0, j1 = cs0.opposite()
        e,  k0 = cs1.opposite()
        e0, k1 = cs2.opposite()
        assert c == c0 and d == d0 and e == e0
        assert i0 % 2 == j1 % 2 == 1 and i0 == (i1 + 1) % 4

        # get crossings outside the triangle
        C0, I0 = cs0.next().opposite()
        C1, I1 = cs1.next().opposite()
        D0, J0 = cs2.next().opposite()
        D1, J1 = cs0.opposite().next().opposite()
        E0, K0 = cs1.opposite().next().opposite()
        E1, K1 = cs2.opposite().next().opposite()

        # form new adjacencies
        # Rotating the triangle 180°, inner adjacency of c, d, e remains the same.
        # If an outside crossing is actually inside, then we only have to modify adjacencies of vertices of the hexagon
        # 1 distance away, because even distances are impossible, and a 3-distance adjacency remains the same.
        if C0 not in (c, d, e): d[j1 + 2] = C0[I0]
        elif C0 == c:           d[j1 + 2] = e[k0 + 2]
        if C1 not in (c, d, e): e[k0 + 2] = C1[I1]
        elif C1 == d:           e[k0 + 2] = e[k1 + 2]
        if D0 not in (c, d, e): e[k1 + 2] = D0[J0]
        elif D0 == d:           e[k1 + 2] = c[i0 + 2]
        if D1 not in (c, d, e): c[i0 + 2] = D1[J1]
        elif D1 == e:           c[i0 + 2] = c[i1 + 2]
        if E0 not in (c, d, e): c[i1 + 2] = E0[K0]
        elif E0 == e:           c[i1 + 2] = d[j0 + 2]
        if E1 not in (c, d, e): d[j0 + 2] = E1[K1]
        elif E1 == c:           d[j0 + 2] = d[j1 + 2]

        # return orientation of strands
        return k0 % 2 == 0

    def orient(self, *entries: tuple):
        """
        Calculates the number of positive and negative crossings from an orientation.
        * entries: a list of CrossingStrands, one from each component, that defines the orientation
        """
        if len(entries) == 0:
            if any(isinstance(crossing, Crossing) for crossing in self.crossings):
                entries = ((self.crossings[0], 0),)
            else:
                self.n_plus = self.n_minus = 0
                return
        entry_points = {crossing: set() for crossing in self.crossings}
        for cs in entries:
            start = cs = CrossingStrand(*cs)
            while True:
                entry_points[cs.crossing].add(cs.strand_index % 4)
                cs = cs.next().opposite()
                if cs == start:
                    break
        self.n_plus = self.n_minus = 0
        for crossing in self.crossings:
            e = entry_points[crossing]
            if len(e) == 2:
                i, j = e
                if (i + j) % 4 == 1:
                    self.n_minus += 1
                elif (i + j) % 4 == 3:
                    self.n_plus += 1
                else:
                    raise ValueError("Component oriented in both directions.")
            elif len(e) > 2:
                raise ValueError("Component oriented in both directions.")
            elif isinstance(crossing, Crossing):
                raise ValueError("Component not oriented.")

    def get_signs(self):
        """
        Returns n_plus, n_minus, if possible.
        """
        try:
            return self.n_plus, self.n_minus
        except:
            try:
                self.orient()
                return self.n_plus, self.n_minus
            except:
                raise Exception("Link has more than one component. Orient the link first with Link.orient(cs0, cs1, ...)")

    def number_of_states(self, h: int, q: int):
        """
        Returns the number of generators of the Khovanov chain complex CKh in grading (h, q).
        """
        # TODO this is super slow, should be able to give a faster estimate
        from scipy.special import binom
        n_plus, n_minus = self.get_signs()
        n = n_plus + n_minus
        k = h + n_minus
        v = h + n_plus - n_minus - q
        if k < 0:
            return 0
        num = 0
        for ones in combinations(range(n), k):
            smoothing = [1 if i in ones else 0 for i in range(n)]
            SL = SmoothLink(self, smoothing)
            l = len(SL.loops)           # number of loops
            if l >= abs(v) and (l - v) % 2 == 0:
                l_1 = (l - v) // 2      # number of 1-labelled loops
                num += binom(l, l_1)
        return round(num)

    def differential_matrix(self, h: int, q: int, printing=True, row_index=None):
        """
        Returns the matrix of the differential in grading (h, q) -> (h + 1, q), and the dicts
        row_index, column_index, giving which row/column corresponds to a LabelledSmoothing data.
        """
        global time, scipy
        import scipy.sparse

        assert all(isinstance(c, Crossing) for c in self.crossings)     # Strands not handled correctly yet

        if printing:
            import time
            start_time = time.time()
            print("number of columns:", self.number_of_states(h, q))
            print(time.time() - start_time, "calculating matrix size")
            start_time = time.time()

        n_plus, n_minus = self.get_signs()
        n = n_plus + n_minus                # number of crossings
        k = h + n_minus                     # number of 1-smoothed crossings
        v = h + n_plus - n_minus - q        # number of x-labelled minus number of 1-labelled loops (v_minus - v_plus)

        if row_index is None:
            row_index = {}                  # dict {LS_data: row index}
        column_index = {}
        data = []                           # data for sparse array: element
        data_row = []                       # data for sparse array: row index
        data_column = []                    # data for sparse array: column index

        if h < -n_minus or h > n_plus:
            return scipy.sparse.csc_array((len(row_index), 0)), {}, {}

        if printing:
            start = time.time()
        j = 0                               # current column
        crossing_strands = [CrossingStrand(c, i) for c in self.crossings for i in (0, 2) if isinstance(c, Crossing)]
        # generate all SmoothLinks
        for zero_smoothings in combinations(range(n), n - k):
            smoothing = [0 if i in zero_smoothings else 1 for i in range(n)]
            SL = SmoothLink(self, smoothing, make_set=True)
            l = len(SL.loops)               # number of loops
            if l >= abs(v) and (l - v) % 2 == 0:
                # loops (identified by indices) in the order of crossing_strands
                loop_ids = [SL.loops.index(SL.loop_of[cs]) for cs in crossing_strands]
                l_1 = (l - v) // 2          # number of 1-labelled loops
                # generate all states
                for one_labels in combinations(range(l), l_1):
                    label_data = ["1" if loop_id in one_labels else "x" for loop_id in loop_ids]
                    last = -1               # lastly considered crossing index, for determining sign
                    sign = 1                # sign of the states in differential
                    # calculate differential of the state
                    for c in zero_smoothings:
                        # get sign
                        if (c - last) % 2 == 0:
                            sign *= -1
                        last = c
                        # get smoothing_data
                        smoothing_data = smoothing.copy()
                        smoothing_data[c] = 1
                        # get new label_datas
                        label_data_0 = label_data.copy()
                        label_datas = (label_data_0, )
                        if loop_ids[2*c] == loop_ids[2*c+1]:
                            if label_data[2*c] == "1":
                                # 1 -> (1, x) + (x, 1)
                                label_data_1 = label_data_0.copy()
                                label_datas = (label_data_0, label_data_1)
                                split_loop = SL.loop_of[crossing_strands[2*c]]
                                i0 = split_loop.index(crossing_strands[2*c])
                                i1 = split_loop.index(crossing_strands[2*c+1])
                                i0, i1 = min(i0, i1), max(i0, i1)
                                ori = split_loop[i0 + 1] == split_loop[i0].next_in_smoothing(smoothing[c])
                                for index, cs in enumerate(crossing_strands):
                                    try:
                                        i = split_loop.index(cs)
                                        # we use that the CSs appear in the loop of SL.loops in order
                                        if (ori and i0 < i <= i1) or (not ori and i0 <= i < i1):
                                            label_data_0[index] = "x"
                                        else:
                                            label_data_1[index] = "x"
                                    except:
                                        pass
                            else:
                                # x -> (x, x)
                                pass
                        else:
                            if label_data[2*c] == label_data[2*c+1] == "1":
                                # (1, 1) -> 1
                                pass
                            elif label_data[2*c] == label_data[2*c+1] == "x":
                                # (x, x) -> 0
                                continue
                            else:
                                # (1, x), (x, 1) -> x
                                if label_data[2*c] == "1":
                                    relabel_loop = SL.loop_of[crossing_strands[2*c]]
                                else:
                                    relabel_loop = SL.loop_of[crossing_strands[2*c+1]]
                                for index, cs in enumerate(crossing_strands):
                                    if cs in relabel_loop:
                                        label_data_0[index] = "x"

                        # add data to sparse matrix
                        for label_data_i in label_datas:
                            state_data = tuple(smoothing_data + label_data_i)
                            try:
                                i = row_index[state_data]
                            except:
                                i = len(row_index)
                                row_index[state_data] = i
                            data.append(sign)
                            data_row.append(i)
                            data_column.append(j)

                    # moving to the next column
                    column_data = tuple(smoothing + label_data)
                    column_index[column_data] = j
                    j += 1
                    if printing and j % 10000 == 0:
                        print(len(row_index), "x", j)
        if printing:
            print(len(row_index), "x", j)
            print(time.time() - start, "created differential matrix")

        matrix = scipy.sparse.csc_array((data, (data_row, data_column)), (len(row_index), j))
        return matrix, row_index, column_index

    @needs_sage
    def homology_with_generators(self, h: int, q: int, chain_level=False):
        """
        When used within Sage, calculates the Khovanov homology of the link in grading (h, q).
        Viewing homology as a direct sum of cyclic groups, returns generators in the form [(group, CKhElement), ...].
        If chain_level is set to True, returns (homology, chain_generators, differential_matrix) instead.
        """
        # calculate the two differential matrices around (h, q)
        differentials = {}
        chain_generators = {}
        column_index = None
        for hh in (h, h - 1):
            diff_matrix, _, column_index = self.differential_matrix(hh, q, row_index=column_index, printing=False)
            differentials[hh] = matrix(diff_matrix.todense())
            if hh == h:
                chain_generators = column_index

        # Sage's homology calculation
        C = ChainComplex(differentials, base_ring=ZZ)
        homology = C.homology(h, generators=True)
        if isinstance(homology, tuple):
            # weird Sage thing where empty ChainComplex returns (0, ()) homology instead of []
            homology = []
    
        if chain_level:
            return homology, chain_generators, diff_matrix.todense()
        else:
            # formatting the generators into CKhElements
            homology_generators = []
            for group, chain in homology:
                coefficients = chain.vector(h)
                states = []
                for LS_data, column in chain_generators.items():
                    LS = self.LS_data_to_class(LS_data)
                    LS.coefficient = coefficients[column]
                    if LS.coefficient != 0:
                        states.append(LS)
                homology_generators.append((group, CKhElement(states)))
            return homology_generators


class SmoothLink:
    """
    A link with a given smoothing at each crossing.
    * smoothing: a dictionary assigning 0 or 1 to each Crossing.
    * loops: the loops of the link according to the smoothing. Each loop is given as a tuple of
             CrossingStrands (two at each Crossing / Strand) as one goes along the loop.
    * loop_of: a dictionary for fast lookup of the loop containing a CrossingStrand.

    Input parameters:
    * link: the underlying Link.
    * smoothing_list: a list of 0s and 1s in the order of link.crossings.
    """

    __slots__ = ("link", "smoothing", "loops", "loop_of")

    def __init__(self, link: Link = None, smoothing_list: list[int] = [], make_set=False):
        if link is None:
            link = Link()
        self.link = link
        crossings = [c for c in link.crossings if isinstance(c, Crossing)]
        # this uses the id of Crossing objects for the dict's hash function as default, so good.
        self.smoothing = {crossing: smoothing_list[i] for i, crossing in enumerate(crossings)}
        self.loops = []
        self.loop_of = {}
        self.get_loops(make_set)

    def get_loops(self, make_set=True):
        """
        Keeping the loops we already have in self.loops, extends self.loops to all loops.
        """
        # crossing_strands = self.link.crossing_strands(make_set=not first_run)
        crossing_strands = self.link.crossing_strands(make_set=make_set, ignore=self.loop_of)
        for loop in self.loops:
            for cs in loop:
                if cs in crossing_strands:
                    crossing_strands.remove(cs)
        while len(crossing_strands):
            if not make_set:
                loop = [crossing_strands.pop(0)]
            else:
                loop = [crossing_strands.pop()]
            while True:
                last = loop[-1]
                if isinstance(last.crossing, Crossing):
                    next = last.next_in_smoothing(self.smoothing[last.crossing])
                else:
                    next = last.next()
                loop.append(next)
                crossing_strands.remove(next)
                opposite = next.opposite()
                if opposite == loop[0]:
                    self.add_loop(tuple(loop))  # we make it hashable by converting to tuple
                    break
                else:
                    crossing_strands.remove(opposite)
                    loop.append(opposite)

    def remove_loop(self, loop):
        """
        Removes loop from self.loops and self.loop_of.
        """
        self.loops.remove(loop)
        for cs in loop:
            del self.loop_of[cs]

    def add_loop(self, loop):
        """
        Adds loop to self.loops and self.loop_of.
        """
        self.loops.append(loop)
        for cs in loop:
            # if cs in self.loop_of:
            #     raise Exception
            self.loop_of[cs] = loop

    def add_loops_for(self, *crossing_strands):
        """
        Adds loops for the specified crossing_strands.
        """
        # TODO This was an attempt to make loop creations slightly faster, but adding parameters to
        # calls of add_loops_for(...) haven't been finished.
        if crossing_strands == ():
            self.get_loops()
        else:
            for cs in crossing_strands:
                if cs in self.loop_of:
                    continue
                loop = [cs]
                while True:
                    last = loop[-1]
                    if isinstance(last.crossing, Crossing):
                        next = last.next_in_smoothing(self.smoothing[last.crossing])
                    else:
                        next = last.next()
                    loop.append(next)
                    opposite = next.opposite()
                    if opposite == loop[0]:
                        self.add_loop(tuple(loop))  # we make it hashable by converting to tuple
                        break
                    else:
                        loop.append(opposite)

    def __eq__(self, other):
        """
        Returns whether two SmoothLinks are the same instance.
        This is a hard check, so that {state.smooth_link for state in self} gives all
        distinct SmoothLink instances in a CKhElement, even if some have identical values.
        """
        return id(self) == id(other)

    def __hash__(self):
        # We make it hashable for use in dict. This way dict just uses ==.
        return 0

    def __repr__(self):
        """
        Returns a string representation, used by print(self).
        """
        result = str([self.smoothing[c] for c in self.link.crossings if isinstance(c, Crossing)])
        for loop in self.loops:
            result += "\n" + str(loop)
        return result

    def copy(self):
        """
        Returns a new instance of self with original reference to link,
        but new instances of loops, loop_of, and smoothing.
        """
        new = copy.copy(self)
        new.loops = self.loops.copy()
        new.loop_of = self.loop_of.copy()
        new.smoothing = self.smoothing.copy()
        return new

    def mirror(self):
        """
        Returns a mirror of self (i.e. smoothing looks the same, but on the mirrored link).
        """
        new_smoothing = [1 - self.smoothing[c] for c in self.link.crossings]
        return SmoothLink(self.link.mirror(use_stored=True), new_smoothing)

    def morse_birth(self, loop):
        """
        Adds loop to self.loops.
        """
        assert loop not in self.loops
        self.add_loop(loop)
        # (self.smoothing is good as Strands don't need smoothing)

    def morse_death(self, loop):
        """
        Removes loop from self.loops.
        """
        self.remove_loop(loop)
        # (self.smoothing is good as we only remove Strands)

    def morse_saddle(self, cs0: CrossingStrand, cs1: CrossingStrand, quick=False):
        """
        Removes the (1 or 2) loops corresponding to the given CrossingStrands from self.loops,
        and adds the (2 or 1) new loops to the end of self.loops according to the saddle move
        that was performed previously on the adjacency of self.link or on self.smoothing.
        """
        loop_0 = self.loop_of[cs0]
        # cs1 may be on a new strand
        if cs1 in self.loop_of:
            new_strand = False
            loop_1 = self.loop_of[cs1]
        else:
            new_strand = True
            loop_1 = (cs1, cs1.next())
        if loop_0 is not loop_1 and not new_strand:
            self.remove_loop(loop_0)
            self.remove_loop(loop_1)
            if quick:
                self.add_loop(loop_0 + loop_1)  # warning: loops are not continuous after this
            else:
                self.add_loops_for(cs0)
        else:
            self.remove_loop(loop_0)
            self.add_loops_for(cs0, cs1)
        # (self.smoothing is good as the list of Crossings is unchanged)

    def fuse(self, cs: CrossingStrand):
        """
        Updates self.loops according to a Strand.fuse().
        """
        loop = self.loop_of[cs]
        self.remove_loop(loop)
        self.add_loops_for(cs.opposite())

    def reidemeister_1(self, cs: CrossingStrand):
        """
        Updates self.loops and self.smoothing according to the R1 move.
        The modified loop will be moved to the end of self.loops.
        """
        loop_0 = self.loop_of[cs]
        loop_1 = self.loop_of[cs.next()]
        self.remove_loop(loop_0)
        # we either have 1 or 2 different loops
        try:
            self.remove_loop(loop_1)
        except:
            pass
        self.add_loops_for()
        del self.smoothing[cs.crossing]

    def reidemeister_1_up(self, cs: CrossingStrand, cs0: CrossingStrand, positive_crossing: bool):
        """
        Updates self.loops and self.smoothing according to the R1 move.
        The given CrossingStrand cs0 has to be on the new loop.
        """
        if positive_crossing:
            self.smoothing[cs0.crossing] = 0
        else:
            self.smoothing[cs0.crossing] = 1
        # small_loop = (cs0, cs0.opposite())
        # self.loops.append(small_loop)
        self.remove_loop(self.loop_of[cs])
        self.add_loops_for()   # cs, cs0 ?

    def reidemeister_2(self, cs0: CrossingStrand, cs1: CrossingStrand):
        """
        Updates self.loops according to an R2 simplification.
        The modified loop(s) will be moved to the end of self.loops.
        """
        c = cs0.crossing
        d = cs0.opposite().crossing
        # update self.loops
        remove_loops = set()
        for cr, i in ((c, 0), (c, 2), (d, 0), (d, 2)):
            remove_loops.add(self.loop_of[CrossingStrand(cr, i)])
        for loop in remove_loops:
            if loop in self.loops:
                self.remove_loop(loop)
            else:
                raise Exception
        self.add_loops_for()
        # we don't update self.smoothing yet because we will need it in LabelledSmoothing.reidemeister_2

    def reidemeister_2_up(self, cs0: CrossingStrand, cs1: CrossingStrand, c: Crossing, d: Crossing):
        """
        Updates self.loops and self.smoothing according to the R2 move.
        Returns the new SmoothLink.
        """
        self.remove_loop(self.loop_of[cs0])
        try:
            self.remove_loop(self.loop_of[cs1])
        except:
            pass
        new = self.copy()
        self.smoothing[c] = 1
        self.smoothing[d] = 0
        self.add_loops_for()
        new.smoothing[c] = 0
        new.smoothing[d] = 1
        new.add_loops_for()
        return new

    def reidemeister_3(self, cs0: CrossingStrand, cs1: CrossingStrand, cs2: CrossingStrand, positive_orientation: bool):
        """
        Updates self.smoothing and self.loops according to an R3 move.
        Returns the old smoothing and a list of the new SmoothLinks.
        """
        # get old smoothings
        c = cs0.crossing
        d = cs2.crossing
        e = cs1.opposite().crossing
        smoothing = (self.smoothing[c], self.smoothing[d], self.smoothing[e])
        # get new smoothings
        if positive_orientation:
            # Table 5(a) with crossings 1 and 2 swapped in columns
            match smoothing:
                case (0, 0, 0): new_smoothings = ((0, 0, 0),)   # smoothing is same as before
                case (1, 0, 0): new_smoothings = ((0, 1, 0), (0, 0, 1))
                case (0, 1, 0): new_smoothings = ((1, 0, 0),)
                case (0, 0, 1): new_smoothings = ()             # image is zero
                case (1, 1, 0): new_smoothings = ((1, 1, 0), (0, 1, 1))
                case (1, 0, 1): new_smoothings = ((0, 1, 1), (1, 0, 1))
                case (0, 1, 1): new_smoothings = ((0, 1, 1), (1, 0, 1))
                case (1, 1, 1): new_smoothings = ()
        else:
            # Table 5(c) with crossings 1 and 2 swapped in columns
            match smoothing:
                case (0, 0, 0): new_smoothings = ()
                case (1, 0, 0): new_smoothings = ((0, 1, 0), (1, 0, 0))
                case (0, 1, 0): new_smoothings = ((0, 1, 0), (1, 0, 0), (0, 0, 1))
                case (0, 0, 1): new_smoothings = ((0, 0, 1),)   # same as before
                case (1, 1, 0): new_smoothings = ((1, 0, 1),)
                case (1, 0, 1): new_smoothings = ((0, 1, 1),)
                case (0, 1, 1): new_smoothings = ((1, 0, 1),)
                case (1, 1, 1): new_smoothings = ((1, 1, 1),)   # same as before
        # update smoothing
        smooth_links = []
        for i in range(len(new_smoothings)):
            new = self.copy() if i > 0 else self
            new.smoothing[c], new.smoothing[d], new.smoothing[e] = new_smoothings[i]
            smooth_links.append(new)
        # update loops
        for smooth_link in smooth_links:
            cs_list = [cs0, cs0.next(), cs2, cs2.next(), cs1.opposite(), cs1.opposite().next()]
            old_loops = {smooth_link.loop_of[cs] for cs in cs_list}
            for loop in old_loops:
                smooth_link.remove_loop(loop)
            smooth_link.add_loops_for()

        return smoothing, smooth_links[1:]

    def differential(self):
        """
        Returns the list of SmoothLinks appearing in the differential in the order of self.link.crossings.
        """
        result = []
        for c in self.link.crossings:
            if isinstance(c, Crossing) and self.smoothing[c] == 0:
                new = self.copy()
                new.smoothing[c] = 1
                new.morse_saddle(CrossingStrand(c, 0), CrossingStrand(c, 2), quick=True)
                result.append(new)
        return result


class LabelledSmoothing:
    """
    Labelled smoothing (also known as a state) with a coefficient.
    * labels: dictionary assigning the label "1" or "x" to each loop. The loops must be in the form
              given by smooth_link.loops, even after modifications by Morse or Reidemeister moves.
    * coefficient: the coefficient of the state in a linear combination, as will be used in CKhElement.

    Input parameters:
    * smooth_link: the underlying SmoothLink
    * labels_list: a string or list of "1", "x" characters.
    """

    __slots__ = ("smooth_link", "labels", "coefficient")

    def __init__(self, smooth_link: SmoothLink = None, labels_list: str | list = "", coefficient: int = 1):
        if smooth_link is None:
            smooth_link = SmoothLink()  # reassignment, so this doesn't update the default parameter
        self.smooth_link = smooth_link
        if len(smooth_link.loops) != len(labels_list):
            raise ValueError("SmoothLink has %s loops, but %s labels were given." % (len(smooth_link.loops), len(labels_list)))
        self.labels = {loop: labels_list[i] for i, loop in enumerate(smooth_link.loops)}
        self.coefficient = coefficient

    def label_of(self, cs: CrossingStrand):
        """
        Returns the label of a CrossingStrand given by self.labels.
        """
        for loop, label in self.labels.items():
            if cs in loop:
                return label
        raise Exception("Couldn't find label of %s in %s." % (cs, self.labels))

    def ls_loop_of(self, cs: CrossingStrand):
        """
        Returns the loop of a CrossingStrand given by self.labels.
        """
        for loop in self.labels:
            if cs in loop:
                return loop
        raise Exception("Couldn't find loop of %s in %s." % (cs, self.labels))

    def __hash__(self):
        smoothing = [self.smooth_link.smoothing[c] for c in self.smooth_link.link.crossings]
        # labels = [self.label_of(CrossingStrand(c, 0)) for c in self.smooth_link.link.crossings]
        return hash(tuple(smoothing))

    def __eq__(self, other):
        """
        Returns whether the labelled smoothing is the same (the coefficient might be different).
        This is a light check, so that it works well with simplify().
        """
        if self.smooth_link.link != other.smooth_link.link:
            return False
        if self.smooth_link.smoothing != other.smooth_link.smoothing:
            return False
        # self.loops is not given in a canonical way, but smooth_link ensures they're the same,
        # so we only need to check the label of one CrossingStrand on each loop.
        for loop in self.smooth_link.loops:
            try:
                other_label = other.labels[loop]
            except:
                other_label = other.label_of(loop[0])
            if self.labels[loop] != other_label:
                return False
        return True

    def __repr__(self):
        """
        Returns a string representation, used by print(self).
        """
        # TODO should make a better visualization for this
        result = str(self.coefficient) + " * "
        result += str([self.smooth_link.smoothing[c] for c in self.smooth_link.link.crossings if isinstance(c, Crossing)])
        result += "\n"
        if self.labels == {}:
            return result + "  empty link\n"
        for loop, label in self.labels.items():
            result += "  '%s' %s\n" % (label, loop)
        return result

    def print_short(self, end="\n"):
        """
        Prints a short 1-line representation of the labels of loops.
        """
        print(self.coefficient, "*", "".join(self.labels.values()), end=end)

    def copy(self):
        """
        Returns a new instance of self with original references to link and smooth_link,
        but new instances of labels and coefficient.
        """
        new = copy.copy(self)
        new.labels = self.labels.copy()
        return new

    def mirror(self, smooth_link: SmoothLink):
        """
        Returns the mirror of self and swaps 1 <-> x labels.
        """
        new_state = self.copy()
        new_state.smooth_link = smooth_link # self.smooth_link.mirror()
        new_state.labels = {}
        for loop, label in self.labels.items():
            cs = new_state.smooth_link.link.find_cs(loop[0])
            new_loop = new_state.smooth_link.loop_of[cs + 1]
            new_state.labels[new_loop] = "1" if label == "x" else "x"
        return new_state

    def to_data(self):
        """
        Returns data that uniquely identifies self, e.g. (0, 1, "1", "x", "1", "x").
        The first n elements of the tuple are the smoothings of the crossings,
        the next 2n elements are the labels of the CrossingStrands c[0], c[2].
        """
        crossings = self.smooth_link.link.crossings
        assert all(isinstance(c, Crossing) for c in crossings)      # Strands not handled correctly yet
        smoothing_list = [self.smooth_link.smoothing[c] for c in crossings]
        label_list = [self.label_of(c[i]) for c in crossings for i in (0, 2)]
        return tuple(smoothing_list + label_list)

    def morse_birth(self, loop):
        """
        Sets self.labels and self.coefficient for the morse birth of loop.
        """
        self.labels[loop] = "1"
        # (state.coefficient remains the same)

    def morse_death(self, loop):
        """
        Sets self.labels and self.coefficient for the morse death of loop.
        """
        label = self.labels[loop]
        if label == "1":
            self.coefficient = 0
        else:
            assert label == "x"
            # (self.coefficient remains the same)
        del self.labels[loop]

    def morse_saddle(self, loop_0, loop_1, new_loop_0, new_loop_1):
        """
        Sets the labels and coefficients for the morse saddle move specified by the loops.
        Either loop_0 and loop_1 must be the same, or new_loop_0 and new_loop_1.
        Returns the new state if the image has more than one element.
        """
        if loop_0 != loop_1 and new_loop_0 == new_loop_1:
            labels = (self.labels[loop_0], self.labels[loop_1])
            if labels == ("x", "x"):
                self.coefficient = 0
                # we don't update self, so have to be careful to remove it in simplify()
            else:
                if labels == ("1", "x") or labels == ("x", "1"):
                    new_label = "x"
                else:
                    assert labels == ("1", "1")
                    new_label = "1"
                # update self.labels
                del self.labels[loop_0]
                del self.labels[loop_1]
                self.labels[new_loop_0] = new_label

        else:
            assert loop_0 == loop_1 and new_loop_0 != new_loop_1
            # update self.labels
            label = self.labels[loop_0]
            del self.labels[loop_0]
            if label == "1":
                self.labels[new_loop_0] = "1"
                self.labels[new_loop_1] = "x"
                new_state = self.copy()
                new_state.labels[new_loop_0] = "x"
                new_state.labels[new_loop_1] = "1"
                return new_state
            else:
                assert label == "x"
                self.labels[new_loop_0] = "x"
                self.labels[new_loop_1] = "x"

    def dot_half(self, loop):
        """
        Sets self.labels and self.coefficient for 1/2 of a dot decoration of an arc (i.e. two saddle moves).
        This kills an x-labelled loop, and sends a 1-labelled loop to an x-labelled loop.
        """
        if self.labels[loop] == "x":
            self.coefficient = 0
        else:
            assert self.labels[loop] == "1"
            self.labels[loop] = "x"

    def fuse(self, cs: CrossingStrand):
        """
        Updates self.labels according to a Strand.fuse().
        """
        old_loop = self.ls_loop_of(cs)
        new_loop = self.smooth_link.loops[-1]
        self.labels[new_loop] = self.labels[old_loop]
        del self.labels[old_loop]

    def reidemeister_1(self, cs: CrossingStrand, positive_crossing: bool):
        """
        Sets the labels and coefficients for untwisting a R1 loop specified by a CrossingStrand.
        Returns the new state if the image has more than one element.
        """
        # bad smoothing is when there's only one loop around the Crossing
        other_cs = cs.next()
        small_loop = self.ls_loop_of(cs)
        old_loop = self.ls_loop_of(other_cs)
        if small_loop == old_loop:
            self.coefficient = 0
            return

        # replace the Crossing-loop with the Strand-loop in self.labels
        new_loop = self.smooth_link.loops[-1]
        self.labels[new_loop] = self.labels[old_loop]
        del self.labels[old_loop]

        # good smoothing with positive crossing: morse_death
        if positive_crossing:
            self.morse_death(small_loop)

        # good smoothing with negative crossing: 1/2*(dot+death - dot there, death)
        # TODO one of these states is always zero so could simplify
        else:
            new_state = self.copy()
            self.dot_half(small_loop)
            self.morse_death(small_loop)
            new_state.dot_half(new_loop)
            new_state.morse_death(small_loop)
            new_state.coefficient = -new_state.coefficient
            return new_state

    def reidemeister_1_up(self, cs: CrossingStrand, cs0: CrossingStrand, positive_crossing: bool):
        """
        Sets the labels and coefficients for adding an R1 loop.
        cs is next to the small loop, cs0 is on the small loop.
        Returns the new state if there is one.
        """
        # add the new Crossing on the loop in self.labels
        old_loop = self.ls_loop_of(cs)
        new_loop = self.smooth_link.loop_of[cs]
        self.labels[new_loop] = self.labels[old_loop]
        del self.labels[old_loop]

        # negative crossing: morse_birth
        small_loop = self.smooth_link.loop_of[cs0]
        self.morse_birth(small_loop)

        # positive crossing: 1/2*(birth+dot - dot there, birth)
        if positive_crossing:
            new_state = self.copy()
            self.dot_half(small_loop)
            new_state.dot_half(new_loop)
            new_state.coefficient *= -1
            return new_state
        else:
            return

    def reidemeister_2(self, cs0: CrossingStrand, cs1: CrossingStrand, c, d, cs00, cs2):
        """
        Updates labels and coefficients for a simplifying R2 move.
        Returns the new state if the image has more than one element.
        """
        # bad smoothing is when crossings have the same smoothing
        sc, sd = self.smooth_link.smoothing[c], self.smooth_link.smoothing[d]
        if sc == sd:
            self.coefficient = 0
            return

        # get loop representations
        if cs1 == cs0 + 1:
            cs0, cs1 = cs1, cs0
        old_loop_0 = self.ls_loop_of(cs0.next())
        old_loop_1 = self.ls_loop_of(cs1.opposite().next())
        new_loop_0 = self.smooth_link.loop_of[cs00]
        new_loop_1 = self.smooth_link.loop_of[cs2]

        # small loop in the middle: -(death, saddle)
        if cs0.next_in_smoothing(sc) == cs1:
            self.coefficient = -self.coefficient
            small_loop = self.ls_loop_of(cs0)
            self.morse_death(small_loop)
            new_state = self.morse_saddle(old_loop_0, old_loop_1, new_loop_0, new_loop_1)
            return new_state

        # horizontal smoothing: labels remain the same
        else:
            self.labels[new_loop_0] = self.labels[old_loop_0]
            del self.labels[old_loop_0]
            if old_loop_1 in self.labels:
                self.labels[new_loop_1] = self.labels[old_loop_1]
                del self.labels[old_loop_1]

    def reidemeister_2_up(self, cs0, cs1, d, smooth_link: SmoothLink):
        """
        Sets the labels and coefficients for adding two Crossings by an R2 move.
        Returns the new states.
        """
        # make a copy of self for the second smoothing
        new_state = self.copy()
        new_state.smooth_link = smooth_link

        # small loop in the middle: (birth, saddle)
        old_loop_0 = self.ls_loop_of(cs0)
        old_loop_1 = self.ls_loop_of(cs1)
        new_loop_0 = self.smooth_link.loop_of[cs0]
        new_loop_1 = self.smooth_link.loop_of[CrossingStrand(d, 2)]
        small_loop = self.smooth_link.loop_of[CrossingStrand(d, 0)]
        self.morse_birth(small_loop)
        new = self.morse_saddle(old_loop_0, old_loop_1, new_loop_0, new_loop_1)

        # horizontal smoothing: labels remain the same
        new_loop_0 = smooth_link.loop_of[cs0]
        new_loop_1 = smooth_link.loop_of[cs1]
        new_state.labels[new_loop_0] = new_state.labels[old_loop_0]
        del new_state.labels[old_loop_0]
        if old_loop_1 in new_state.labels:
            new_state.labels[new_loop_1] = new_state.labels[old_loop_1]
            del new_state.labels[old_loop_1]

        if new:
            return [new, new_state]
        else:
            return [new_state]

    def reidemeister_3(
        self,
        cs0: CrossingStrand,
        cs1: CrossingStrand,
        cs2: CrossingStrand,
        positive_orientation: bool,
        smoothing: tuple,
        smooth_links: list[SmoothLink],
    ):
        """
        Updates labels and coefficients for an R3 move.
        Returns the new states if the image has more than one element.
        """
        states: list[LabelledSmoothing] = []
        for smooth_link in smooth_links:
            new_state = self.copy()
            new_state.smooth_link = smooth_link
            states.append(new_state)

        # get loops
        cs_old = [cs0.next(),            cs1.opposite().next(), cs2.next()           ]      # these always correspond to three different strands
        cs_new = [cs0.opposite().next(), cs1.next(),            cs2.opposite().next()]
        old_loops = [self.ls_loop_of(cs) for cs in cs_old]

        # functions that will be used a lot in the Tables
        def identity(state, sign=None):
            new_loops = [state.smooth_link.loop_of[cs] for cs in cs_new]
            for old_loop, new_loop in zip(old_loops, new_loops):
                assert old_loop not in new_loops  # I think it can't happen that we get the same old_loop back after an R3 move, so ok.
                if old_loop in state.labels:
                    state.labels[new_loop] = state.labels[old_loop]
                    del state.labels[old_loop]
            if sign is not None:
                state.coefficient *= sign

        def saddle(state, indices, birth=False, death=False, sign=None):
            def fix_loop(state, i):
                if new_loops[i] not in state.labels:
                    state.labels[new_loops[i]] = state.labels[old_loops[i]]
                    del state.labels[old_loops[i]]

            if death:
                old_small_loop = state.ls_loop_of(cs0)
                state.morse_death(old_small_loop)
                if state.coefficient == 0:
                    return
            if birth:
                small_loop = state.smooth_link.loop_of[cs0]
                state.morse_birth(small_loop)
            if sign is not None:
                state.coefficient *= sign
            if len(indices) == 2:
                i, j = indices
                new_loops = [state.smooth_link.loop_of[cs] for cs in cs_new]
                new_state = state.morse_saddle(old_loops[i], old_loops[j], new_loops[i], new_loops[j])
                k = 3 - i - j
                fix_loop(state, k)
                if new_state:
                    fix_loop(new_state, k)
                    states.append(new_state)
            elif indices == [0, 1, 2]:
                # using a temporary loop because we're doing two saddle moves in one step
                temp_loop = tuple()
                new_loops = [state.smooth_link.loop_of[cs] for cs in cs_new]
                # first saddle move
                intermediate_states = [state]
                if old_loops[0] != old_loops[1]:
                    new_state = state.morse_saddle(old_loops[0], old_loops[1], temp_loop, temp_loop)
                else:
                    new_state = state.morse_saddle(old_loops[0], old_loops[0], new_loops[0], temp_loop)
                if state.coefficient == 0:      # if morse_saddle gives 0, LS might be trash
                    return
                if new_state:
                    intermediate_states.append(new_state)
                # second saddle move
                new_states = []
                for intermediate_state in intermediate_states:
                    if new_loops[1] == new_loops[2]:
                        new_state = intermediate_state.morse_saddle(temp_loop, old_loops[2], new_loops[1], new_loops[1])
                    else:
                        new_state = intermediate_state.morse_saddle(temp_loop, temp_loop, new_loops[1], new_loops[2])
                    if new_state:
                        new_states.append(new_state)
                for new_state in intermediate_states[1:] + new_states:
                    states.append(new_state)
                # states += intermediate_states[1:] + new_states                          # this would cause bugs I think
            else:
                raise ValueError(indices)

        # Table 5(a)
        if positive_orientation:
            match smoothing:
                case (0, 0, 0):
                    identity(self)
                case (1, 0, 0):
                    identity(self)
                    identity(states[0])
                case (0, 1, 0):
                    identity(self)
                case (0, 0, 1):
                    self.coefficient = 0
                case (1, 1, 0):
                    identity(self, sign=-1)     # extra sign because crossings 1 and 2 are swapped here
                    saddle(states[0], [1, 2], birth=True)
                case (1, 0, 1):
                    saddle(self, [0, 1], birth=True, sign=-1)
                    identity(states[0], sign=-1)
                case (0, 1, 1):
                    saddle(self, [0, 1, 2], birth=True, death=True)
                    saddle(states[0], [0, 2], death=True)
                case (1, 1, 1):
                    self.coefficient = 0
        # Table 5(c)
        else:
            match smoothing:
                case (0, 0, 0):
                    self.coefficient = 0
                case (1, 0, 0):
                    saddle(self, [0, 1], birth=True)
                    identity(states[0])
                case (0, 1, 0):
                    saddle(self, [0, 1, 2], birth=True, death=True, sign=-1)
                    saddle(states[0], [0, 2], death=True, sign=-1)
                    saddle(states[1], [1, 2], death=True, sign=-1)
                case (0, 0, 1):
                    identity(self)
                case (1, 1, 0):
                    identity(self)
                case (1, 0, 1):
                    identity(self)
                case (0, 1, 1):
                    identity(self)
                case (1, 1, 1):
                    identity(self, sign=-1)     # extra sign because crossings 1 and 2 are swapped here

        if len(states) > 0:
            return states

    def differential(self, smooth_links: list[SmoothLink] = None):
        """
        Returns the differential of self as a list of LabelledSmoothings.
        """
        if smooth_links is None:
            smooth_links = self.smooth_link.differential()
        result = []
        parity = 0
        i = 0
        for c in self.smooth_link.link.crossings:
            if isinstance(c, Crossing):
                if self.smooth_link.smoothing[c] == 0:
                    new = self.copy()
                    new.smooth_link = smooth_links[i]
                    if parity == 1:
                        new.coefficient *= -1
                    cs0 = CrossingStrand(c, 0)
                    cs1 = CrossingStrand(c, 2)
                    loop_0 = new.ls_loop_of(cs0)
                    loop_1 = new.ls_loop_of(cs1)
                    new_loop_0 = new.smooth_link.loop_of[cs0]
                    new_loop_1 = new.smooth_link.loop_of[cs1]
                    new_state = new.morse_saddle(loop_0, loop_1, new_loop_0, new_loop_1)
                    if new.coefficient != 0:
                        result.append(new)
                    if new_state and new_state.coefficient != 0:
                        result.append(new_state)
                    i += 1
                else:
                    parity = 1 - parity
        return result
    
    def is_outside(self):
        """
        Checks if self is outside the states appearing in the image of the differential map,
        i.e. if self.row_size() is zero.
        Equivalently, if every 1-smoothed crossing connects two different 1-labelled loops.
        """
        for c in self.smooth_link.link.crossings:
            if self.smooth_link.smoothing[c] == 1:
                cs0, cs2 = CrossingStrand(c, 0), CrossingStrand(c, 2)
                if self.smooth_link.loop_of[cs0] == self.smooth_link.loop_of[cs2]:
                    return False
                if self.label_of(cs0) == "x" or self.label_of(cs2) == "x":
                    return False
        return True

    def row_size(self):
        """
        Returns the number of nonzero elements in the row corresponding to self in the 
        differential matrix, i.e. len(self.reverse_differential()).
        """
        total = 0
        for c in self.smooth_link.link.crossings:
            if self.smooth_link.smoothing[c] == 1:
                cs0, cs2 = CrossingStrand(c, 0), CrossingStrand(c, 2)
                if self.smooth_link.loop_of[cs0] == self.smooth_link.loop_of[cs2]:
                    if self.label_of(cs0) == "1":
                        total += 1
                    else:
                        total += 2
                else:
                    if self.label_of(cs0) == "x" or self.label_of(cs2) == "x":
                        total += 1
        return total
    
    def reverse_differential(self):
        """
        Returns the list of LabelledSmoothings whose differential contains self.
        """
        flipped = self.copy()
        for loop, label in flipped.labels.items():
            flipped.labels[loop] = "1" if label == "x" else "x"

        new_smooth_links = []
        for c in flipped.smooth_link.link.crossings:
            if isinstance(c, Crossing) and flipped.smooth_link.smoothing[c] == 1:
                new = flipped.smooth_link.copy()
                new.smoothing[c] = 0
                new.morse_saddle(CrossingStrand(c, 0), CrossingStrand(c, 2), quick=True)
                new_smooth_links.append(new)

        result = []
        i = 0
        for c in flipped.smooth_link.link.crossings:
            if isinstance(c, Crossing):
                if flipped.smooth_link.smoothing[c] == 1:
                    new = flipped.copy()
                    new.smooth_link = new_smooth_links[i]
                    cs0 = CrossingStrand(c, 0)
                    cs1 = CrossingStrand(c, 2)
                    loop_0 = new.ls_loop_of(cs0)
                    loop_1 = new.ls_loop_of(cs1)
                    new_loop_0 = new.smooth_link.loop_of[cs0]
                    new_loop_1 = new.smooth_link.loop_of[cs1]
                    new_state = new.morse_saddle(loop_0, loop_1, new_loop_0, new_loop_1)
                    if new.coefficient != 0:
                        result.append(new)
                    if new_state and new_state.coefficient != 0:
                        result.append(new_state)
                    i += 1

        for state in result:
            for loop, label in state.labels.items():
                state.labels[loop] = "1" if label == "x" else "x"
        return result

    def minimal_row_size(self):
        """
        Returns the minimal row_size() of nonzero elements in the column corresponding to self
        in the differential matrix.
        """
        return min(state.row_size() for state in self.differential())


class CKhElement(list[LabelledSmoothing]):
    """
    A linear combination of states, given as a list of LabelledSmoothings.
    We allow different states to reference the same instance of a SmoothLink.
    """

    __slots__ = ()

    def simplify(self):
        """
        Modifies the CKhElement in place by combining identical states and removing zero elements.
        """
        # TODO could do this faster without tmpstates

        # combine like terms
        tmpstates = []
        while len(self) != 0:
            state = self.pop(0)
            if state.coefficient != 0:
                # we need to remove 0-coefficient elements here too,
                # because their smooth_link might be trash
                if state not in tmpstates:
                    tmpstates.append(state)
                else:
                    i = tmpstates.index(state)
                    tmpstates[i].coefficient += state.coefficient

        # remove 0 elements
        if self != []:
            raise Exception("Error in simplify.")
        for state in tmpstates:
            if state.coefficient != 0:
                self.append(state)

    def link(self):
        """
        Returns the underlying link.
        """
        if len(self) > 0:
            return self[0].smooth_link.link

    def __repr__(self):
        """
        Returns a string representation, used by print(self).
        """
        if len(self) == 0:
            return "0\n"
        result = ""
        for i in range(len(self)):
            result += str(self[i])
            if i < len(self) - 1:
                result += "+ "
        return result

    def print_short(self):
        """
        Prints self in a shorter way than print(self), e.g. 1 * 1xx1x1 + 1 * xx11x1 + -1 * xx1x11.
        """
        if len(self) == 0:
            print("0")
        for i in range(len(self)):
            self[i].print_short(end="")
            if i < len(self) - 1:
                print(" + ", end="")
            else:
                print()

    def print_long(self, printing={"long"}):
        """
        Prints self in a customizable way.
        The printing value can be "short", "long", "length", "link", "differential", or any set of these.
        """
        if "length" in printing:
            print("length:", len(self))
        if "link" in printing and len(self) > 0:
            print(self.link())
        if "long" in printing:
            print(self)
        if "short" in printing:
            self.print_short()
        if "differential" in printing:
            print("differential:", len(self.differential()))

    def morse_birth(self):
        """
        Does a Morse birth on Link, SmoothLinks, and LabelledSmoothings.
        Returns a CrossingStrand of the new Strand.
        """
        if len(self) == 0:
            return

        # add new loop to link
        link = self.link()
        loop = link.morse_birth()

        # add new loop to each state.smooth_link
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.morse_birth(loop)

        # add new loop to each state.labels
        for state in self:
            state.morse_birth(loop)

        # (there's no reason to self.simplify())
        return loop[0]

    def morse_death(self, cs: CrossingStrand | tuple):
        """
        Does a Morse death specified by a CrossingStrand on Link, SmoothLinks, and LabelledSmoothings.
        """
        if len(self) == 0:
            return
        if not isinstance(cs, CrossingStrand):
            cs = CrossingStrand(*cs)
        assert cs.crossing.is_loop()
        loop = self[0].ls_loop_of(cs)

        # remove loop from link
        link = self.link()
        link.morse_death(cs)

        # remove loop from each state.smooth_link
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.morse_death(loop)

        # remove loop from each state.labels
        for state in self:
            state.morse_death(loop)

        # TODO just removing 0 coefficient elements would be less calculation
        self.simplify()

    def morse_saddle(self, cs0: CrossingStrand | tuple, cs1: CrossingStrand | tuple):
        """
        Does a Morse saddle move specified by the two CrossingStrands on Link, SmoothLinks, and LabelledSmoothings.
        """
        if len(self) == 0:
            return
        if not isinstance(cs0, CrossingStrand):
            cs0 = CrossingStrand(*cs0)
        if not isinstance(cs1, CrossingStrand):
            cs1 = CrossingStrand(*cs1)

        # update crossing adjacency of link
        link = self.link()
        cs0_opp = link.morse_saddle(cs0, cs1)

        # update loops for each state.smooth_link
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.morse_saddle(cs1, cs0_opp)

        # update labels and coefficients for each state and get new states
        new_states = []
        for state in self:
            loop_0 = state.ls_loop_of(cs0)
            loop_1 = state.ls_loop_of(cs1)
            if loop_0 == loop_1:
                new_loop_0, new_loop_1 = state.smooth_link.loops[-2:]
            else:
                new_loop_0 = new_loop_1 = state.smooth_link.loops[-1]
            new_state = state.morse_saddle(loop_0, loop_1, new_loop_0, new_loop_1)
            if new_state:
                new_states.append(new_state)
        for new_state in new_states:
            self.append(new_state)

        self.simplify()

    def fuse(self, strand: Strand):
        """
        Fuses a Strand on Link, SmoothLinks, and LabelledSmothings.
        """
        if len(self) == 0:
            return

        # update crossings in link
        link = self.link()
        link.fuse(strand)

        # update loops in each state.smooth_link
        cs = CrossingStrand(strand, 0)
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.fuse(cs)

        # update labels for each state
        for state in self:
            state.fuse(cs)

        # there's no reason to self.simplify()

    def reidemeister_1(self, cs: CrossingStrand | tuple):
        """
        Untwists a Reidemeister-1 loop specified by a CrossingStrand on Link, SmoothLinks, and LabelledSmothings.
        Fuses the new Strand if possible, otherwise returns the Strand.
        """
        if len(self) == 0:
            return
        if not isinstance(cs, CrossingStrand):
            cs = CrossingStrand(*cs)

        # move the relevant crossing to the end, need this for correct signs
        new_order = [x for x in self.link().crossings if x != cs.crossing] + [cs.crossing]
        self.reorder_crossings(new_order)

        # remove R1 loop from link
        link = self.link()
        positive_crossing, s = link.reidemeister_1(cs)

        # remove R1 loop from each state.smooth_link
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.reidemeister_1(cs)

        # update labels and coefficients for each state and get new states
        new_states = []
        for state in self:
            new_state = state.reidemeister_1(cs, positive_crossing)
            if new_state:
                new_states.append(new_state)
        for new_state in new_states:
            self.append(new_state)

        self.simplify()
        return s

    def reidemeister_1_up(self, cs: CrossingStrand | tuple, positive_crossing: bool):
        """
        Adds a Reidemeister-1 loop to the left of a CrossingStrand to Link, SmoothLinks, and LabelledSmothings.
        Returns the CrossingStrand on the left of the new loop.
        """
        if len(self) == 0:
            return
        if not isinstance(cs, CrossingStrand):
            cs = CrossingStrand(*cs)

        # add R1 loop to link
        link = self.link()
        cs0 = link.reidemeister_1_up(cs, positive_crossing)

        # add R1 loop to each smooth_link
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.reidemeister_1_up(cs, cs0, positive_crossing)

        # update labels and coefficients for each state and get new states
        new_states = []
        for state in self:
            new_state = state.reidemeister_1_up(cs, cs0, positive_crossing)
            if new_state:
                new_states.append(new_state)
        for new_state in new_states:
            self.append(new_state)

        self.simplify()

        return cs0

    def reidemeister_2(self, cs0: CrossingStrand | tuple, cs1: CrossingStrand | tuple):
        """
        Does a simplifying Reidemeister-2 move specified by the CrossingStrands on Link, SmoothLinks, and LabelledSmothings.
        Returns two CrossingStrands on the two strands.
        """
        if len(self) == 0:
            return
        if not isinstance(cs0, CrossingStrand):
            cs0 = CrossingStrand(*cs0)
        if not isinstance(cs1, CrossingStrand):
            cs1 = CrossingStrand(*cs1)

        # move the relevant crossings to the end, need this for correct signs
        c, d = cs0.crossing, cs0.opposite().crossing
        new_order = [x for x in self.link().crossings if x not in (c, d)] + [c, d]
        self.reorder_crossings(new_order)

        # R2 simplification on link
        link = self.link()
        c, d, cs2, cs3 = link.reidemeister_2(cs0, cs1)

        # update each smooth_link.loops
        for smooth_link in {state.smooth_link for state in self}:
            smooth_link.reidemeister_2(cs0, cs1)

        # update labels and coefficients for each state and get new states
        new_states = []
        for state in self:
            new_state = state.reidemeister_2(cs0, cs1, c, d, cs2, cs3)
            if new_state:
                new_states.append(new_state)
        for new_state in new_states:
            self.append(new_state)

        # update each smooth_link.smoothing
        for smooth_link in {state.smooth_link for state in self}:
            del smooth_link.smoothing[c]
            del smooth_link.smoothing[d]

        self.simplify()

        return cs2, cs3

    def reidemeister_2_up(self, cs0: CrossingStrand | tuple, cs1: CrossingStrand | tuple):
        """
        Adds two Crossings by a Reidemeister-2 move to Link, SmoothLinks, and LabelledSmothings.
        Returns the two CrossingStrands in the middle left.
        """
        if len(self) == 0:
            return
        if not isinstance(cs0, CrossingStrand):
            cs0 = CrossingStrand(*cs0)
        if not isinstance(cs1, CrossingStrand):
            cs1 = CrossingStrand(*cs1)

        # R2 move on link
        link = self.link()
        c, d = link.reidemeister_2_up(cs0, cs1)

        # R2 move on each smooth_link
        new_smooth_link = {}
        for smooth_link in {state.smooth_link for state in self}:
            new_smooth_link[smooth_link] = smooth_link.reidemeister_2_up(cs0, cs1, c, d)

        # update labels and coefficients for each state and get new states
        new_states = []
        for state in self:
            new_state_list = state.reidemeister_2_up(cs0, cs1, d, new_smooth_link[state.smooth_link])
            if new_state_list:
                for new_state in new_state_list:
                    new_states.append(new_state)
        for new_state in new_states:
            self.append(new_state)

        self.simplify()

        return CrossingStrand(c, 1), CrossingStrand(c, 2)

    def reidemeister_3(self, cs0: CrossingStrand | tuple, cs1: CrossingStrand | tuple, cs2: CrossingStrand | tuple):
        """
        Does a Reidemeister-3 move specified by the CrossingStrands on Link, SmoothLink, and LabelledSmoothings.
        """
        if len(self) == 0:
            return
        if not isinstance(cs0, CrossingStrand):
            cs0 = CrossingStrand(*cs0)
        if not isinstance(cs1, CrossingStrand):
            cs1 = CrossingStrand(*cs1)
        if not isinstance(cs2, CrossingStrand):
            cs2 = CrossingStrand(*cs2)

        # move the relevant crossings to the end, need this for correct signs
        c, d, e = cs0.crossing, cs2.crossing, cs1.opposite().crossing
        new_order = [x for x in self.link().crossings if x not in (c, d, e)] + [c, d, e]
        self.reorder_crossings(new_order)

        # R3 move on link
        link = self.link()
        positive_orientation = link.reidemeister_3(cs0, cs1, cs2)

        # R3 move on each smooth_link
        data = {}
        for smooth_link in {state.smooth_link for state in self}:
            old_smoothing, new_smooth_links = smooth_link.reidemeister_3(cs0, cs1, cs2, positive_orientation)
            data[smooth_link] = old_smoothing, new_smooth_links

        # update labels and coefficients for each state and get new states
        new_states = []
        for state in self:
            new_state_list = state.reidemeister_3(cs0, cs1, cs2, positive_orientation, *data[state.smooth_link])
            if new_state_list:
                for new_state in new_state_list:
                    new_states.append(new_state)
        for new_state in new_states:
            self.append(new_state)

        self.simplify()

    def differential(self, simplify=True):
        """
        Returns the differential of self.
        """
        if len(self) == 0:
            return CKhElement()

        if len(self) > 1:
            # do each saddle move on each state.smooth_link
            data = {}
            for smooth_link in {state.smooth_link for state in self}:
                new_smooth_links = smooth_link.differential()
                data[smooth_link] = new_smooth_links
            # do each saddle move on each state
            result = []
            for state in self:
                new_states = state.differential(data[state.smooth_link])
                for new_state in new_states:
                    if new_state.coefficient != 0:
                        result.append(new_state)
        else:
            # do each saddle move on smooth_link
            new_smooth_links = self[0].smooth_link.differential()
            # do each saddle move
            result = self[0].differential(new_smooth_links)

        result = CKhElement(result)
        if simplify:
            result.simplify()
        return result

    def replace_link(self, link: Link | None = None, flipping: list[Crossing | Strand] = []):
        """
        Replaces the underlying link_0 by link via a bijection of crossings given in the order of
        link_0.crossings and link.crossings and rotating by 180° the crossings of the new link in flipping.
        If no new link is given, then it changes the adjacency of link_0 according to flipping.
        """

        def map(cs0: CrossingStrand):
            if link is not None:
                c0, i0 = cs0
                index = crossings_0.index(c0)
                c1 = crossings_1[index]
                if c1 not in flipping:
                    i1 = i0
                elif isinstance(c1, Crossing):
                    i1 = (i0 + 2) % 4
                elif isinstance(c1, Strand):
                    i1 = (i0 + 1) % 2
                return CrossingStrand(c1, i1)
            else:
                if cs0.crossing not in flipping:
                    return cs0
                else:
                    return cs0 + 2

        if len(self) == 0:
            return

        if link is not None:
            # check that link is the same as link_0 via map()
            link_0 = self.link()
            crossings_0 = link_0.crossings.copy()
            crossings_1 = link.crossings
            for cs0 in link_0.crossing_strands():
                if map(cs0.opposite()) != map(cs0).opposite():
                    print(f"Adjacency {cs0} {cs0.opposite()} doesn't correspond to {map(cs0)} {map(cs0).opposite()}.")

            # replace link
            # link_0.crossings = link.crossings

            # update link, smoothing and loops on each smooth_link
            for smooth_link in {state.smooth_link for state in self}:
                smooth_link.link = link
                new_dict = {}
                for c0, c1 in zip(crossings_0, crossings_1):
                    new_dict[c1] = smooth_link.smoothing[c0]
                smooth_link.smoothing = new_dict
                smooth_link.loops = []
                smooth_link.loop_of = {}
                smooth_link.add_loops_for()

            # update labels on each state
            for state in self:
                new_labels = {}
                for loop, label in state.labels.items():
                    new_loop = state.smooth_link.loop_of[map(loop[0])]
                    new_labels[new_loop] = label
                state.labels = new_labels

        else:
            # change link
            for crossing in flipping:
                crossing.flip()

            # update loops on each smooth_link
            for smooth_link in {state.smooth_link for state in self}:
                smooth_link.loops = []
                smooth_link.loop_of = {}
                smooth_link.add_loops_for()

            # update labels on each state
            for state in self:
                new_labels = {}
                for loop, label in state.labels.items():
                    new_loop = state.smooth_link.loop_of[map(loop[0])]
                    new_labels[new_loop] = label
                state.labels = new_labels

    def reorder_crossings(self, new_order: list[Crossing | Strand]):
        """
        Reorder the crossings of link. This changes some of the signs of the coefficients.
        """
        if len(self) == 0:
            return

        # update link
        link = self.link()
        old_order = link.crossings
        link.crossings = new_order  # this is an assignment, so old_order is unchanged
        assert len(new_order) == len(old_order) and set(new_order) == set(old_order)

        # have to multiply those states by (-1) where the permutation restricted to the 1-smoothed crossings is odd
        odd = set()
        for smooth_link in {state.smooth_link for state in self}:
            ones = [c for c in old_order if isinstance(c, Crossing) and smooth_link.smoothing[c] == 1]
            parity = 0
            for i in range(len(ones)):
                for j in range(i + 1, len(ones)):
                    i_new = new_order.index(ones[i])
                    j_new = new_order.index(ones[j])
                    if i_new > j_new:
                        parity = 1 - parity
            if parity == 1:
                odd.add(smooth_link)

        # update states
        for state in self:
            if state.smooth_link in odd:
                state.coefficient *= -1

    def copy(self):
        """
        Returns a new instance of self with original references to link and smooth_link,
        but new instance of each state.
        """
        new = copy.copy(self)
        for i in range(len(new)):
            new[i] = copy.copy(new[i])
        return new

    def mirror(self):
        """
        Returns the mirror of self and swaps labels 1 <-> x.
        """
        if len(self) == 0:
            return CKhElement()

        # create link.mirror_link
        self.link().mirror()

        # mirror each smooth_link
        mirror = {}
        for smooth_link in {state.smooth_link for state in self}:
            mirror[smooth_link] = smooth_link.mirror()

        # mirror each state
        new_states = []
        for state in self:
            new_state = state.mirror(mirror[state.smooth_link])
            new_states.append(new_state)
        
        return CKhElement(new_states)

    def __neg__(self):
        """
        Negates the coefficients of a CKhElement. Return value has the original
        references to Link, SmoothLinks, but new instance of LabelledSmoothings.
        """
        states = []
        for state in self:
            new_state = state.copy()
            new_state.coefficient *= -1
            states.append(new_state)
        return CKhElement(states)

    def __add__(self, other):
        """
        Adds together two CKhElements of the same Link. Return value has the original
        references to Link, SmoothLinks, but new instance of LabelledSmoothings.
        """
        total = CKhElement(list.__add__(self.copy(), other.copy()))
        total.simplify()
        return total

    def __sub__(self, other):
        """
        Subtracts a CKhElement from another of the same Link. Return value has the original
        references to Link, SmoothLinks, but new instance of LabelledSmoothings.
        """
        total = CKhElement(list.__add__(self.copy(), -other))
        total.simplify()
        return total


class Cobordism:
    """
    A class for handling cobordisms between two links conveniently.

    Input a cobordism by giving the starting link and specifying the elementary moves in order,
    or use finish() if the last link is the unlink. For example:

    >>> L = Link([a, b, c])
    >>> C = Cobordism(L)
    >>> C.morse_saddle((a, 0), (a, 1))
    >>> C.reidemeister_1((a, 0))
    >>> C.finish()

    Properties:
    * links: the list of links corresponding to each stage of a movie of the cobordism
    * movie: list of elementary moves (movie[i] is between links[i] and links[i + 1]),
             given as (function_name, arguments, return_value)
    """
    def __init__(self, starting_link: Link = None):
        if starting_link == None:
            starting_link = Link()
        self.links = [starting_link]
        self.movie = []
        self.rename = None

    def __repr__(self):
        """
        Returns the moves of the movie as a string, used by print(self).
        """
        text = ""
        for move in self.movie:
            text += str(move) + "\n"
        if len(self.movie) == 0:
            text = "0-length movie"
        return text

    def print_long(self):
        """
        Prints the movie with all the intermediate links.
        """
        for link, move in zip(self.links, self.movie):
            print(link, "\n", move, "\n")
        print(self.links[-1])
        if self.rename:
            print("\n", self.rename)

    def copy(self):
        """
        Returns a copy of self with new instances of links and movie, but original
        references to the Link objects and CrossingStrands.
        """
        new = Cobordism()
        new.links = self.links.copy()
        new.movie = self.movie.copy()
        # new.rename = self.rename
        return new

    def chi(self):
        """
        Returns the Euler characteristic of the cobordism, calculated from its Morse moves.
        """
        chi = 0
        for move in self.movie:
            match move[0]:
                case "morse_birth" | "morse_death":
                    chi += 1
                case "morse_saddle":
                    chi -= 1
        return chi
    
    @needs_sage
    def matrix(self, h: int, q: int):
        """
        When used within Sage, calculates the cobordism map induced on homology in matrix form.
        Returns ((start_group, end_group), matrix).
        """
        # calculate homology for the starting and ending link
        start_homology_generators = self.links[0].homology_with_generators(h, q)   
        end_homology, end_chain_generators, differential_matrix = self.links[-1].homology_with_generators(h, q + self.chi(), True)
        start_group = [group for group, _ in start_homology_generators]
        end_group = [group for group, _ in end_homology]
        
        if len(start_group) == 0 or len(end_group) == 0:
            return (start_group, end_group), matrix([])

        # calculate the images of generators under the cobordism
        images = []
        for _, generator in start_homology_generators:
            gen_copy = copy.deepcopy(generator)
            self.map(gen_copy)
            image = [0] * len(end_chain_generators)
            for state in gen_copy:
                i = end_chain_generators[state.to_data()]
                image[i] = state.coefficient
            images.append(image)
        images = matrix(images).transpose()

        # solve linear equation
        if len(end_homology) > 0:
            # add generators of homology as new columns to the differential matrix
            vectors = [v.vector(h) for (group, v) in end_homology]
            matrix_and_basis = matrix(list(differential_matrix.transpose()) + vectors).transpose()
            solution = matrix_and_basis.solve_right(images)
            solution = solution[-len(end_homology):]
        else:
            solution = matrix([])

        return (start_group, end_group), solution

    def move(self, function_name: str, *arguments):
        """
        Adds the elementary move function_name(*arguments) to the end of the cobordism.
        The CrossingStrands in the arguments will be changed to be from the correct Link.
        """
        new_link = self.links[-1].copy()
        new_arguments = []
        arguments_fixed = []
        for obj in arguments:
            if isinstance(obj, CrossingStrand):
                new_arguments.append(new_link.find_cs(obj))
                arguments_fixed.append(self.links[-1].find_cs(obj))
            elif isinstance(obj, Strand):
                new_arguments.append(new_link.find_crossing(obj))
                arguments_fixed.append(self.links[-1].find_crossing(obj))
            else:
                new_arguments.append(obj)
                arguments_fixed.append(obj)
        return_value = getattr(new_link, function_name)(*new_arguments)
        self.movie.append((function_name, arguments_fixed, return_value))
        self.links.append(new_link)
        assert len(new_link.crossings) == len(set([c.label for c in new_link.crossings]))  # check that labels are unique
        return return_value

    def morse_birth(self):
        """
        Adds a Morse birth move to the movie.
        """
        new_loop = self.move("morse_birth")
        return new_loop[0]

    def morse_death(self, cs: tuple):
        """
        Adds a Morse death move to the movie.
        """
        self.move("morse_death", CrossingStrand(*cs))

    def morse_saddle(self, cs0: tuple, cs1: tuple):
        """
        Adds a Morse saddle move to the movie.
        """
        self.move("morse_saddle", CrossingStrand(*cs0), CrossingStrand(*cs1))

    def fuse(self, s: Strand):
        """
        Adds a fuse move to the movie.
        """
        self.move("fuse", s)

    def reidemeister_1(self, cs: tuple):
        """
        Adds a Reidemeister 1 move to the movie.
        """
        _, s = self.move("reidemeister_1", CrossingStrand(*cs))
        return s

    def reidemeister_1_up(self, cs: tuple, positive_crossing: bool):
        """
        Adds a Reidemeister 1 move to the movie, increasing the crossing number.
        """
        return self.move("reidemeister_1_up", CrossingStrand(*cs), positive_crossing)

    def reidemeister_2(self, cs0: tuple, cs1: tuple):
        """
        Adds a Reidemeister 2 move to the movie.
        """
        _, _, cs2, cs3 = self.move("reidemeister_2", CrossingStrand(*cs0), CrossingStrand(*cs1))
        return cs2, cs3

    def reidemeister_2_up(self, cs0: tuple, cs1: tuple):
        """
        Adds a Reidemeister 2 move to the movie, increasing the crossing number.
        """
        c, _ = self.move("reidemeister_2_up", CrossingStrand(*cs0), CrossingStrand(*cs1))
        return CrossingStrand(c, 1), CrossingStrand(c, 2)

    def reidemeister_3(self, cs0: tuple, cs1: tuple, cs2: tuple):
        """
        Adds a Reidemeister 3 move to the movie.
        """
        self.move("reidemeister_3", CrossingStrand(*cs0), CrossingStrand(*cs1), CrossingStrand(*cs2))

    def try_r1_r2(self):
        """
        Tries to do a Reidemeister 1, Reidemeister 2, Morse death, or fuse move
        on the last Link. Returns True if succeeded.
        """
        link = self.links[-1]
        # try reidemeister_1 at all Crossings
        for c in link.crossings:
            if isinstance(c, Crossing):
                for i in range(4):
                    if c.adjacent[i] == (c, (i + 1) % 4):
                        self.move("reidemeister_1", CrossingStrand(c, i))
                        return True
        # try reidemeister_2 at all Crossings
        for c in link.crossings:
            if isinstance(c, Crossing):
                for i in range(4):
                    (d, j), (e, k) = c.adjacent[i], c.adjacent[(i + 1) % 4]
                    if d == e and isinstance(d, Crossing) and (j - 1) % 4 == k and (i + j) % 2 == 0:
                        self.move("reidemeister_2", CrossingStrand(c, i), CrossingStrand(c, (i + 1) % 4))
                        return True
        # try morse_death / fuse at all Strands
        for s in link.crossings:
            if isinstance(s, Strand):
                if s.is_loop():
                    self.move("morse_death", CrossingStrand(s, 0))
                else:
                    self.move("fuse", s)
                return True
        return False
    
    def possible_r3_moves(self):
        """
        Returns the list of possible Reidemeister 3 moves on the last Link.
        """
        link = self.links[-1]
        moves = []
        for c in link.crossings:
            if isinstance(c, Crossing):
                for i in range(4):
                    (d, j), (e, k) = c.adjacent[i], c.adjacent[(i - 1) % 4]
                    if i % 2 == 1 and j % 2 == 1 and d.adjacent[(j + 1) % 4] == (e, (k - 1) % 4):
                        moves.append((CrossingStrand(c, i),
                                      CrossingStrand(c, (i - 1) % 4),
                                      CrossingStrand(d, (j + 1) % 4)))
                        assert len({c, d, e}) == 3
        return moves
    
    def finish(self, max_closest=100, max_random=100):
        """
        Keeps doing simplifying Reidemeister 1, 2, 3 moves and Morse deaths until no longer possible.
        It tries to make the sequence of moves as short as possible by first checking the neighbours of
        the closest max_closest nodes in the tree of possible R3 moves for R1 or R2 simplification,
        and if unsuccessful, does the same with max_random random R3 moves (similar to Snappy's "level"
        simplification).
        """
        # do R1 and R2 simplifications until none are possible
        while self.try_r1_r2():
            pass

        # try to find the shortest R3 move sequence to fewer crossings by exploring the full tree
        tree = [self]
        i = 0
        while i < max_closest and i < len(tree):
            cobordism = tree[i]
            for crossing_strands in cobordism.possible_r3_moves():
                new_cobordism = cobordism.copy()
                new_cobordism.move("reidemeister_3", *crossing_strands)
                if new_cobordism.try_r1_r2():
                    # found an R1 or R2 simplification, reset the tree to start from there
                    while new_cobordism.try_r1_r2():
                        pass
                    tree = [new_cobordism]
                    i = 0
                    break
                else:
                    # neighbour can't be simplified, add it to the tree to explore further
                    tree.append(new_cobordism)
            else:
                i += 1

        cobordism = tree[0]
        num_moves = len(cobordism.movie)

        # after max_tries tries, stop exploring the full tree and just do random R3 moves
        if i == max_closest:
            from random import choice as random_choice
            if cobordism is self:
                cobordism = cobordism.copy()
            failures = 0
            while failures < max_random:
                possible_moves = cobordism.possible_r3_moves()
                if len(possible_moves) == 0:
                    break
                cobordism.move("reidemeister_3", *random_choice(possible_moves))
                if cobordism.try_r1_r2():
                    # found and R1 or R2 simplification, reset failures count
                    while cobordism.try_r1_r2():
                        pass
                    num_moves = len(cobordism.movie)
                    failures = 0
                else:
                    failures += 1

        # copy the relevant data into self
        for function_name, arguments, return_value in cobordism.movie[len(self.movie) : num_moves]:
            self.move(function_name, *arguments)

    def band_move(self, twists, *crossing_strands):
        """
        Does a band move across two strands that are not necessarily in the same region of the diagram.
        The band is specified by the CrossingStrands that the left side of the band crosses.
        Note: right now, the band can't cross itself or the same strand multiple times.

        * crossing_strands=[cs0, (cs1, True), (cs2, False), ..., csk] means the band starts
        at cs0, goes over cs1, goes under cs2, etc. and ends at csk.
        * twists is the number of half-twists of the band.
        """
        #  cs0  cs1  cs2                csk
        #   |    |    |           3      |
        #   \---------|--   --\ /\ /\ /--/
        #        |    |  ...   \  \  \
        #   /---------|--   --/ \/ \/ \--\
        #   |    |    |                  |

        cs_current = self.links[-1].find_cs(CrossingStrand(*crossing_strands[0]))
        cs_last = self.links[-1].find_cs(CrossingStrand(*crossing_strands[-1]))
        for cs_i, over in crossing_strands[1 : -1]:
            cs_i = self.links[-1].find_cs(CrossingStrand(*cs_i))
            if over:
                cs2, _ = self.reidemeister_2_up(cs_current.opposite(), cs_i.opposite())
                cs_current = cs2.opposite()
            else:
                _, cs_current = self.reidemeister_2_up(cs_i, cs_current)

        for i in range(abs(twists)):
            cs_current = self.reidemeister_1_up(cs_current, twists < 0)
        self.morse_saddle(cs_current, cs_last)

    def mirror(self):
        """
        Returns the mirror of the cobordism.
        """
        M = Cobordism(self.links[0].mirror(use_stored=True))
        rename = {}  # has the form {label: (Crossing, int)}, telling us the Crossing in the mirror and how much we have to rotate
        # the Crossing is not necessarily from the correct Link instance, but it has good label
        for c in self.links[0].crossings:
            rename[c.label] = c, 1 if isinstance(c, Crossing) else 0

        def map(obj: CrossingStrand | Strand):
            # maps a CS (or Strand) of self to the corresponding CS (or Strand) in the mirror
            if isinstance(obj, CrossingStrand):
                return CrossingStrand(*rename[obj.crossing.label]) + obj.strand_index
            elif isinstance(obj, Strand):
                return rename[obj.label][0]

        for function_name, arguments, return_value in self.movie:
            match function_name:
                case "morse_birth":
                    loop = M.move(function_name)
                    label = return_value[0].crossing.label
                    rename[label] = loop[0].crossing, 0
                case "morse_death" | "fuse":
                    M.move(function_name, map(arguments[0]))
                case "morse_saddle":
                    cs2 = M.move(function_name, map(arguments[0]), map(arguments[1]))
                    s = return_value.crossing
                    if isinstance(s, Strand) and s.is_loop():       # TODO this looks risky, assumes that there were no fusable Strands
                        rename[s.label] = cs2.crossing, 0
                case "reidemeister_1":
                    s = return_value[1]
                    _, s0 = M.move(function_name, map(arguments[0]))
                    if s:
                        rename[s.label] = s0, 0
                case "reidemeister_2":
                    s2, s3 = return_value[2].crossing, return_value[3].crossing
                    _, _, cs20, cs30 = M.move(function_name, map(arguments[0]), map(arguments[1]))
                    if isinstance(s2, Strand) and s2.is_loop():     # this looks risky
                        rename[s2.label] = cs20.crossing, 0
                    if s3 != s2 and isinstance(s3, Strand) and s3.is_loop():
                        rename[s3.label] = cs30.crossing, 0
                case "reidemeister_1_up":
                    cs, positive_crossing = arguments
                    cs0 = M.move(function_name, map(cs), not positive_crossing)
                    label = return_value.crossing.label
                    rename[label] = cs0.crossing, -1 if positive_crossing else 1
                case "reidemeister_2_up":
                    cs0, cs1 = arguments
                    c, d = return_value
                    d0, c0 = M.move(function_name, map(cs1.opposite()), map(cs0.opposite()))
                    rename[c.label] = c0, -1
                    rename[d.label] = d0, 1
                case "reidemeister_3":
                    cs0, cs1, cs2 = arguments
                    positive_orientation = return_value
                    if positive_orientation:
                        M.move(function_name, map(cs1.opposite()), map(cs2.opposite()), map(cs0))
                    else:
                        M.move(function_name, map(cs2), map(cs0.opposite()), map(cs1.opposite()))

        # if self.rename is None:
        #     M.rename = {c.label: rename[c.label] for c in self.links[-1].crossings}
        # else:
        #     original_name = {c: (label, 4-x) for label, (c, x) in self.rename.items()}
        #     M.rename = {self.rename[c.label]: rename[c.label] for c in self.links[-1].crossings}

        M.rename = {c.label: rename[c.label] for c in self.links[-1].crossings}
        return M

    def reverse(self):
        """
        Returns the reverse of the cobordism.
        """
        R = Cobordism(self.links[-1])
        rename = {}  # has the form {label: (Crossing, int)}, telling us the Crossing in the reverse and how much we have to rotate
        for c in R.links[0].crossings:
            rename[c.label] = c, 0

        def map(obj: CrossingStrand | Strand):
            # maps a CS of self to the corresponding CS in the reverse
            if isinstance(obj, CrossingStrand):
                return CrossingStrand(*rename[obj.crossing.label]) + obj.strand_index
            elif isinstance(obj, Strand):
                return rename[obj.label][0]

        movie = self.movie.copy()
        movie.reverse()
        for function_name, arguments, return_value in movie:
            match function_name:
                case "morse_birth":
                    cs = return_value[0]
                    R.move("morse_death", map(cs))
                case "morse_death":
                    loop = R.move("morse_birth")
                    label = arguments[0].crossing.label
                    rename[label] = loop[0].crossing, 0
                case "morse_saddle":
                    cs0 = arguments[0]
                    cs2 = return_value
                    R.move("morse_saddle", map(cs0), map(cs2))
                    s = cs2.crossing
                    if isinstance(s, Strand) and s.is_loop():
                        R.move("fuse", map(s))
                case "fuse":
                    raise Exception         # Reverse of fuse not yet implemented, but shouldn't be hard
                case "reidemeister_1":
                    cs0 = arguments[0]
                    positive_crossing, s = return_value
                    cs1 = cs0.opposite()
                    assert cs0.crossing == cs1.crossing
                    if cs0 + 1 == cs1:
                        cs0, cs1 = cs1, cs0
                    cs = CrossingStrand(s, 0) if s else (cs1 + 2).opposite()
                    cs00 = R.move("reidemeister_1_up", map(cs), positive_crossing)
                    label = cs0.crossing.label
                    rename[label] = cs00.crossing, 0 if cs0.strand_index in (2, 3) else 2
                    if s:
                        R.move("fuse", map(s))
                case "reidemeister_1_up":
                    cs = return_value
                    R.move("reidemeister_1", map(cs))
                case "reidemeister_2":
                    cs0, cs1 = arguments
                    c, d, cs2, cs3 = return_value
                    if cs0 + 1 == cs1:
                        cs0, cs1 = cs1, cs0
                    if cs0.strand_index % 2 == 0:
                        c0, d0 = R.move("reidemeister_2_up", map(cs3.opposite()), map(cs2))
                        rename[c.label] = c0, 0 if cs0.strand_index == 2 else 2
                        rename[d.label] = d0, 0 if cs0.opposite().strand_index == 0 else 2
                    else:
                        d0, c0 = R.move("reidemeister_2_up", map(cs2.opposite()), map(cs3))
                        rename[c.label] = c0, 0 if cs0.strand_index == 1 else 2
                        rename[d.label] = d0, 0 if cs0.opposite().strand_index == 1 else 2
                    s2, s3 = cs2.crossing, cs3.crossing
                    if isinstance(s2, Strand) and s2.is_loop():
                        R.move("fuse", map(s2))
                    if s3 != s2 and isinstance(s3, Strand) and s3.is_loop():
                        R.move("fuse", map(s3))
                case "reidemeister_2_up":
                    c = return_value[0]
                    cs0, cs1 = CrossingStrand(c, 1), CrossingStrand(c, 2)
                    R.move("reidemeister_2", map(cs0), map(cs1))
                case "reidemeister_3":
                    cs0, cs1, cs2 = arguments
                    R.move("reidemeister_3", map(cs0), map(cs1), map(cs2))

        for L, RL in ((self.links[0], R.links[-1]), (self.links[-1], R.links[0])):
            if hasattr(L, "n_plus"):
                RL.n_plus, RL.n_minus = L.n_plus, L.n_minus
        
        R.rename = {c.label: rename[c.label] for c in self.links[0].crossings}
        return R

    def map(self, CKH: CKhElement, printing={}):
        """
        Changes a CKhElement to its image under the cobordism.
        """
        if len(CKH) == 0:
            return
        link = CKH.link()
        for function_name, arguments, _ in self.movie:
            # if link has been abandoned, need to stop (otherwise find_cs might give error)
            if len(CKH) == 0:
                break
            # update arguments to the correct Crossing objects
            new_arguments = []
            for obj in arguments:
                if isinstance(obj, CrossingStrand):
                    new_arguments.append(link.find_cs(obj))
                elif isinstance(obj, Strand):
                    new_arguments.append(link.find_crossing(obj))
                else:
                    new_arguments.append(obj)
            # print info, which might involve calculating differential
            CKH.print_long(printing)
            if printing:
                print("------------", function_name, arguments)
            # do the actual move
            getattr(CKH, function_name)(*new_arguments)

        CKH.print_long(printing)

    def KJ_class(self, printing={}):
        """
        Returns the Khovanov-Jacobsson class of a cobordism that has the empty link on one of its ends.
        This is the image of the generator of Z under the cobordism map Z -> CKh(K).
        The printing value can be a bool, "short", "long", "length", "link", "differential", or any set of these.
        """
        if isinstance(printing, str) or printing is True:
            printing = {"length", printing}
        elif printing is False:
            printing = {}
        CKH = CKhElement([LabelledSmoothing()])
        if len(self.links[0].crossings) == 0:
            self.map(CKH, printing)
            # replace link to the original
            CKH.replace_link(self.links[-1])
            return CKH
        if len(self.links[-1].crossings) == 0:
            R = self.reverse()
            R.map(CKH, printing)
            if len(CKH) == 0:
                return CKH
            # replace link to the original, using the rename dict
            new_order = []
            flipping = []
            link = CKH.link()
            for c in self.links[0].crossings:
                r, i = R.rename[c.label]
                new_order.append(link.find_crossing(r))
                if i == 2:
                    flipping.append(c)
            CKH.reorder_crossings(new_order)
            CKH.replace_link(self.links[0], flipping)
            if printing:
                print("------------", "rename", R.rename)
                CKH.print_long(printing)
                print()
            return CKH
        raise Exception("One of the ends of the cobordism has to be the empty link.")


def compare(CKH0: CKhElement, CKH1: CKhElement, printing=True, check_differential=False):
    """
    Calculates and prints whether two CKhElements (in a single grading (h, q)
    of the same Link) are the same in homology up to sign.
    """

    # make sure they have 0 differential and are on the same link
    if check_differential:
        diff = len(CKH0.differential()), len(CKH1.differential())   # TODO write a fast len_differential function
        print("differential:", *diff)
        assert diff == (0, 0)
    if printing:
        print("length:", len(CKH0), len(CKH1))
    if len(CKH0) > 0 and len(CKH1) > 0:
        assert CKH0.link() is CKH1.link()
        CKHs = {"total": CKH0 + CKH1, "difference": CKH0 - CKH1}
        if printing:
            print(f"total: {len(CKHs['total'])}, difference: {len(CKHs['difference'])}")
    elif len(CKH0) > 0 or len(CKH1) > 0:
        CKHs = {"total": CKH0 + CKH1}
        if printing:
            print(f"total: {len(CKHs['total'])}")
    else:
        if printing:
            print("\nBoth chain elements are zero.\n")
        return True

    # check if they are trivially outside the image of the differential map
    for label in ("total", "difference"):
        if label in CKHs and any(state.is_outside() for state in CKHs[label]):
            if printing:
                print(label, "is outside")
            del CKHs[label]
        if label in CKHs and len(CKHs[label]) == 0:
            if printing:
                print(f"\nThe homology classes are the same up to sign ({label} is 0).\n")
            return True
    if len(CKHs) == 0:
        if printing:
            print("\nThe homology classes are different (they have states outside).\n")
        return False

    global time, scipy
    import time
    import scipy.sparse
    import numpy
    
    # calculate n+, n- if L is a knot. If L has more than one component, it has to be oriented previously.
    LS = CKH0[0] if len(CKH0) > 0 else CKH1[0]
    L = LS.smooth_link.link
    n_plus, n_minus = L.get_signs()
    # TODO detect orientation from cobordism saddle moves?

    # calculate grading
    k = sum(LS.smooth_link.smoothing.values())          # number of 1-smoothed crossings
    v_plus = list(LS.labels.values()).count("1")        # number of 1-labelled loops
    v_minus = list(LS.labels.values()).count("x")       # number of x-labelled loops
    h = k - n_minus                                     # homological grading
    q = v_plus - v_minus + h + n_plus - n_minus         # quantum grading
    if printing:
        print(f"(n+, n-) = ({n_plus}, {n_minus})")
        print(f"(h, q) = ({h}, {q})")

    # calculate the matrix of the differential map
    matrix, row_index, _ = L.differential_matrix(h - 1, q, printing=printing)

    # create vectors from the CKhElements
    start_time = time.time()
    r = len(row_index)
    vectors = {}
    for label, CKH in CKHs.items():
        vector = numpy.zeros(r, dtype=int)
        for state in CKH:
            i = row_index[state.to_data()]
            vector[i] = state.coefficient
        vectors[label] = vector
    if printing:
        print(time.time() - start_time, "created vectors")

    # solve linear equations
    start_time = time.time()
    tolerance = 1e-10
    for label, vector in vectors.items():
        solution = scipy.sparse.linalg.lsqr(matrix, vector, atol=tolerance, btol=tolerance)
        if printing:
            print(label, "istop:", solution[1], "r1norm:", solution[3])
        if not solution[1] == 2:
            if printing:
                print(time.time() - start_time, "solved linear equations")
                print(f"\nThe homology classes are the same up to sign ({label} is 0).\n")
            return True
    if printing:
        print(time.time() - start_time, "solved linear equations")
        print("\nThe homology classes are different.\n")
    return False

