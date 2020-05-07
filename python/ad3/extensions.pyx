from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

from base cimport Factor
from base cimport BinaryVariable
from base cimport MultiVariable
from base cimport FactorGraph
from base cimport PBinaryVariable, PMultiVariable, PFactor, PGenericFactor


cdef extern from "../examples/cpp/dense/FactorSequence.h" namespace "AD3":
    cdef cppclass FactorSequence(Factor):
        FactorSequence()
        void Initialize(vector[int] num_states)


cdef extern from "../examples/cpp/summarization/FactorSequenceCompressor.h" namespace "AD3":
    cdef cppclass FactorSequenceCompressor(Factor):
        FactorSequenceCompressor()
        void Initialize(int length, vector[int] left_positions,
                        vector[int] right_positions)


cdef extern from "../examples/cpp/summarization/FactorCompressionBudget.h" namespace "AD3":
    cdef cppclass FactorCompressionBudget(Factor):
        FactorCompressionBudget()
        void Initialize(int length, int budget,
                        vector[bool] counts_for_budget,
                        vector[int] bigram_positions)


cdef extern from "../examples/cpp/summarization/FactorBinaryTree.h" namespace "AD3":
    cdef cppclass FactorBinaryTree(Factor):
        FactorBinaryTree()
        void Initialize(vector[int] parents)


cdef extern from "../examples/cpp/summarization/FactorBinaryTreeCounts.h" namespace "AD3":
    cdef cppclass FactorBinaryTreeCounts(Factor):
        FactorBinaryTreeCounts()
        void Initialize(vector[int] parents, vector[bool] counts_for_budget)
        void Initialize(vector[int] parents, vector[bool] counts_for_budget,
                        vector[bool] has_count_scores)
        void Initialize(vector[int] parents, vector[bool] counts_for_budget,
                        vector[bool] has_count_scores, int max_num_bins)


cdef extern from "../examples/cpp/summarization/FactorGeneralTree.h" namespace "AD3":
    cdef cppclass FactorGeneralTree(Factor):
        FactorGeneralTree()
        void Initialize(vector[int] parents, vector[int] num_states)


cdef extern from "../examples/cpp/summarization/FactorGeneralTreeCounts.h" namespace "AD3":
    cdef cppclass FactorGeneralTreeCounts(Factor):
        FactorGeneralTreeCounts()
        void Initialize(vector[int] parents, vector[int] num_states)


cdef extern from "../examples/cpp/parsing/FactorTree.h" namespace "AD3":
    cdef cppclass Arc:
        Arc(int, int)

    cdef cppclass FactorTree(Factor):
        FactorTree()
        void Initialize(int, vector[Arc *])
        int RunCLE(vector[double]&, vector[int] *v, double *d)


cdef extern from "../examples/cpp/parsing/FactorHeadAutomaton.h" namespace "AD3":
    cdef cppclass Sibling:
        Sibling(int, int, int)

    cdef cppclass FactorHeadAutomaton(Factor):
        FactorHeadAutomaton()
        void Initialize(vector[Arc *], vector[Sibling *])
        void Initialize(int, vector[Sibling *])


cdef extern from "../examples/cpp/parsing/FactorGrandparentHeadAutomaton.h" namespace "AD3":
    cdef cppclass Grandparent:
        Grandparent(int, int, int)

    cdef cppclass Grandsibling:
        Grandsibling(int, int, int, int)

    cdef cppclass FactorGrandparentHeadAutomaton(Factor):
        FactorGrandparentHeadAutomaton()
        void Initialize(vector[Arc*], vector[Arc*], vector[Grandparent*],
            vector[Sibling*])
        void Initialize(vector[Arc*], vector[Arc*], vector[Grandparent*], 
            vector[Sibling*], vector[Grandsibling*])


cdef class PFactorSequence(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequence()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] num_states):
        (<FactorSequence*>self.thisptr).Initialize(num_states)


cdef class PFactorSequenceCompressor(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequenceCompressor()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, vector[int] left_positions,
                   vector[int] right_positions):
        (<FactorSequenceCompressor*>self.thisptr).Initialize(length,
                                                             left_positions,
                                                             right_positions)


cdef class PFactorCompressionBudget(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorCompressionBudget()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, int budget,
                   pcounts_for_budget,
                   vector[int] bigram_positions):
        cdef vector[bool] counts_for_budget
        for counts in pcounts_for_budget:
            counts_for_budget.push_back(counts)
        (<FactorCompressionBudget*>self.thisptr).Initialize(length, budget,
                                                            counts_for_budget,
                                                            bigram_positions)


cdef class PFactorBinaryTree(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinaryTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents):
        (<FactorBinaryTree*>self.thisptr).Initialize(parents)


cdef class PFactorBinaryTreeCounts(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinaryTreeCounts()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents,
                   pcounts_for_budget,
                   phas_count_scores=None,
                   max_num_bins=None):
        cdef vector[bool] counts_for_budget
        cdef vector[bool] has_count_scores
        for counts in pcounts_for_budget:
            counts_for_budget.push_back(counts)
        if phas_count_scores is not None:
            for has_count in phas_count_scores:
                has_count_scores.push_back(has_count)
            if max_num_bins is not None:
                (<FactorBinaryTreeCounts*>self.thisptr).Initialize(
                    parents, counts_for_budget, has_count_scores, max_num_bins)

            else:
                (<FactorBinaryTreeCounts*>self.thisptr).Initialize(
                    parents, counts_for_budget, has_count_scores)

        else:
            (<FactorBinaryTreeCounts*>self.thisptr).Initialize(
                parents, counts_for_budget)


cdef class PFactorGeneralTree(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorGeneralTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents, vector[int] num_states):
        (<FactorGeneralTree*>self.thisptr).Initialize(parents, num_states)


cdef class PFactorGeneralTreeCounts(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorGeneralTreeCounts()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents, vector[int] num_states):
        (<FactorGeneralTreeCounts*>self.thisptr).Initialize(parents,
                                                            num_states)


cdef class PFactorTree(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, list arcs, bool validate=True):
        cdef vector[Arc *] arcs_v
        cdef int head, modifier

        cdef tuple arc
        for arc in arcs:
            head = arc[0]
            modifier = arc[1]

            if validate:
                if not 0 <= head < length:
                    raise ValueError("Invalid arc: head must be in [0, length)")
                if not 1 <= modifier < length:
                    raise ValueError("Invalid arc: modifier must be in ",
                                     "[1, length)")
                if not head != modifier:
                    raise ValueError("Invalid arc: head cannot be the same as "
                                     "the modifier")
            arcs_v.push_back(new Arc(head, modifier))

        if validate and arcs_v.size() != <Py_ssize_t> self.thisptr.Degree():
            raise ValueError("Number of arcs differs from number of bound "
                             "variables.")
        (<FactorTree*>self.thisptr).Initialize(length, arcs_v)

        for arcp in arcs_v:
            del arcp


cdef class PFactorHeadAutomaton(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorHeadAutomaton()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, arcs_or_length, list siblings, bool validate=True):
        """
        If arcs_or_length is the length, it is assumed that all possible arcs
        in the tree exist, i.e., there was no pruning.
        The length is relative to the head position. 
        E.g. for a right automaton with h=3 and instance_length=10,
        length = 7. For a left automaton, it would be length = 3.

        If any arcs were pruned, arcs_or_length should be a list with the
        remaining arcs.
        """
        cdef tuple arc
        cdef vector[Arc *] arcs_v
        cdef vector[Sibling *] siblings_v
        cdef tuple sibling
        cdef int length

        for sibling in siblings:
            siblings_v.push_back(new Sibling(sibling[0],
                                             sibling[1],
                                             sibling[2]))
        
        if isinstance(arcs_or_length, int):
            length = arcs_or_length
        else:
            length = max(s - h for (h, m, s) in siblings)

        if validate:
            if siblings_v.size() != length * (1 + length) / 2:
                raise ValueError("Inconsistent length passed.")

            if length != self.thisptr.Degree() + 1:
                raise ValueError("Number of variables doesn't match.")

        if isinstance(arcs_or_length, list):
            for arc in arcs_or_length:
                arcs_v.push_back(new Arc(arc[0], arc[1]))

            (<FactorHeadAutomaton*>self.thisptr).Initialize(arcs_v, siblings_v)
            for arcp in arcs_v:
                del arcp
        else:
            (<FactorHeadAutomaton*>self.thisptr).Initialize(length, siblings_v)

        for sibp in siblings_v:
            del sibp

cdef class PFactorGrandparentHeadAutomaton(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new FactorGrandparentHeadAutomaton()
    
    def __dealloc__(self):
        if self.allocate:
            del self.thisptr
    
    def initialize(self, list incoming_arcs, list outgoing_arcs, list grandparents,
        list siblings, list grandsiblings=None):
        """
        Incoming arcs are of the form (g,h) for each g.
        Outgoing arcs are of the form (h,m) for each m.
        
        Siblings are tuples (h, m, s)
        Grandparents are tuples (g, h, m)
        Grandsiblings are tuples (g, h, m, s)

        The variables linked to this factor must be in the same order as
        the incoming arcs, followed by the outgoing arcs.

        The incoming arcs must be sorted by grandparent, from smallest to
        biggest. Incoming arcs must be included even if they don't have a corresponding
        grandparent part. This happens when the only sibling parts are (h, h, 0) and
        (h, h, length).

        The outgoing arcs must be sorted from the closest to the farthest
        away from the head.

        Grandparent parts must include all combinations of incoming and outgoing arcs,
        including the tuples in which g = m.

        When solving the problem, the returned additional posteriors will be ordered
        as grandparent factors, then sibling factors, and finally grandsiblings if
        any.

        Potentials should be set in the same order.
        """

        cdef vector[Arc *] incoming_v, outgoing_v
        cdef vector[Sibling *] siblings_v
        cdef vector[Grandparent *] grandparents_v
        cdef vector[Grandsibling *] grandsiblings_v
        
        cdef tuple arc
        cdef tuple sibling
        cdef tuple grandparent
        cdef tuple grandsibling
        for arc in incoming_arcs:
            incoming_v.push_back(new Arc(arc[0], arc[1]))
        
        for arc in outgoing_arcs:
            outgoing_v.push_back(new Arc(arc[0], arc[1]))
        
        for sibling in siblings:
            siblings_v.push_back(new Sibling(sibling[0], 
                                             sibling[1], 
                                             sibling[2]))
        
        for grandparent in grandparents:
            grandparents_v.push_back(new Grandparent(grandparent[0], 
                                                     grandparent[1],
                                                     grandparent[2]))
        
        if grandsiblings is None:
            (<FactorGrandparentHeadAutomaton*>self.thisptr).Initialize(
                incoming_v, outgoing_v, grandparents_v, siblings_v)
        else:
            for grandsibling in grandsiblings:
                grandsiblings_v.push_back(new Grandsibling(grandsibling[0],
                                                           grandsibling[1],
                                                           grandsibling[2],
                                                           grandsibling[3]))
            
            (<FactorGrandparentHeadAutomaton*>self.thisptr).Initialize(
                incoming_v, outgoing_v, grandparents_v, siblings_v, grandsiblings_v)

        for arcp in incoming_v:
            del arcp
        for arcp in outgoing_v:
            del arcp
        for sibp in siblings_v:
            del sibp
        for gpp in grandparents_v:
            del gpp
        for gsp in grandsiblings_v:
            del gsp
