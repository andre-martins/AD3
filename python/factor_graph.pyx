from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

# get the classes from the c++ headers

cdef extern from "../ad3/Factor.h" namespace "AD3":
    cdef cppclass BinaryVariable:
        BinaryVariable()
        double GetLogPotential()
        void SetLogPotential(double log_potential)
        int GetId()
        int Degree()

    cdef cppclass Factor:
        Factor()
        vector[double] GetAdditionalLogPotentials()
        void SetAdditionalLogPotentials(vector[double] additional_log_potentials)
        int Degree()
        int GetLinkId(int i)
        BinaryVariable *GetVariable(int i)
        void SolveMAP(vector[double] variable_log_potentials,
                      vector[double] additional_log_potentials,
                      vector[double] *variable_posteriors,
                      vector[double] *additional_posteriors,
                      double *value)


cdef extern from "../ad3/MultiVariable.h" namespace "AD3":
    cdef cppclass MultiVariable:
        int GetNumStates()
        BinaryVariable *GetState(int i)
        double GetLogPotential(int i)
        void SetLogPotential(int i, double log_potential)


cdef extern from "../ad3/FactorGraph.h" namespace "AD3":
    cdef cppclass FactorGraph:
        FactorGraph()
        void SetVerbosity(int verbosity)
        void SetEtaPSDD(double eta)
        void SetMaxIterationsPSDD(int max_iterations)
        int SolveLPMAPWithPSDD(vector[double]* posteriors,
                               vector[double]* additional_posteriors,
                               double* value)
        void SetEtaAD3(double eta)
        void AdaptEtaAD3(bool adapt)
        void SetMaxIterationsAD3(int max_iterations)
        void SetResidualThresholdAD3(double threshold)
        void FixMultiVariablesWithoutFactors()
        int SolveLPMAPWithAD3(vector[double]* posteriors,
                              vector[double]* additional_posteriors,
                              double* value)
        int SolveExactMAPWithAD3(vector[double]* posteriors,
                                 vector[double]* additional_posteriors,
                                 double* value)

        vector[double] GetDualVariables()
        vector[double] GetLocalPrimalVariables()
        vector[double] GetGlobalPrimalVariables()

        BinaryVariable *CreateBinaryVariable()
        MultiVariable *CreateMultiVariable(int num_states)
        Factor *CreateFactorDense(vector[MultiVariable*] multi_variables,
                                  vector[double] additional_log_potentials,
                                  bool owned_by_graph)
        Factor *CreateFactorXOR(vector[BinaryVariable*] variables,
                                vector[bool] negated,
                                bool owned_by_graph)
        Factor *CreateFactorXOROUT(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   bool owned_by_graph)
        Factor *CreateFactorAtMostOne(vector[BinaryVariable*] variables,
                                      vector[bool] negated,
                                      bool owned_by_graph)
        Factor *CreateFactorOR(vector[BinaryVariable*] variables,
                               vector[bool] negated,
                               bool owned_by_graph)
        Factor *CreateFactorOROUT(vector[BinaryVariable*] variables,
                                  vector[bool] negated,
                                  bool owned_by_graph)
        Factor *CreateFactorANDOUT(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   bool owned_by_graph)
        Factor *CreateFactorIMPLY(vector[BinaryVariable*] variables,
                                  vector[bool] negated,
                                  bool owned_by_graph)
        Factor *CreateFactorPAIR(vector[BinaryVariable*] variables,
                                 double edge_log_potential,
                                 bool owned_by_graph)
        Factor *CreateFactorBUDGET(vector[BinaryVariable*] variables,
                                   vector[bool] negated,
                                   int budget,
                                   bool owned_by_graph)
        Factor *CreateFactorKNAPSACK(vector[BinaryVariable*] variables,
                                     vector[bool] negated,
                                     vector[double] costs,
                                     double budget,
                                     bool owned_by_graph)
        void DeclareFactor(Factor *factor,
                           vector[BinaryVariable*] variables,
                           bool owned_by_graph)


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


# wrap them into python extension types
cdef class PBinaryVariable:
    cdef BinaryVariable *thisptr
    cdef bool allocate
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new BinaryVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def get_log_potential(self):
        return self.thisptr.GetLogPotential()

    def set_log_potential(self, double log_potential):
        self.thisptr.SetLogPotential(log_potential)

    def get_id(self):
        return self.thisptr.GetId()

    def get_degree(self):
        return self.thisptr.Degree()


cdef class PMultiVariable:
    cdef MultiVariable *thisptr
    cdef bool allocate

    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new MultiVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    cdef int _get_n_states(self):
        return self.thisptr.GetNumStates()

    def __len__(self):
        return self._get_n_states()

    def get_state(self, int i, bool validate=True):

        if validate and not 0 <= i < self._get_n_states():
            raise IndexError("State {:d} is out of bounds.".format(i))

        cdef BinaryVariable *variable = self.thisptr.GetState(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def __getitem__(self, int i):

        if not 0 <= i < self._get_n_states():
            raise IndexError("State {:d} is out of bounds.".format(i))

        return self.get_log_potential(i)

    def __setitem__(self, int i, double log_potential):
        if not 0 <= i < len(self):
            raise IndexError("State {:d} is out of bounds.".format(i))
        self.set_log_potential(i, log_potential)

    def get_log_potential(self, int i):
        return self.thisptr.GetLogPotential(i)

    def set_log_potential(self, int i, double log_potential):
        self.thisptr.SetLogPotential(i, log_potential)

    @cython.boundscheck(False)
    def set_log_potentials(self, double[:] log_potentials, bool validate=True):
        cdef Py_ssize_t n_states = self.thisptr.GetNumStates()
        cdef Py_ssize_t i

        if validate and len(log_potentials) != n_states:
            raise IndexError("Expected buffer of length {}".format(n_states))

        for i in range(n_states):
            self.thisptr.SetLogPotential(i, log_potentials[i])


cdef class PFactor:
    cdef Factor* thisptr
    cdef bool allocate
    # This is a virtual class, so don't allocate/deallocate.
    def __cinit__(self):
        self.allocate = False
        pass

    def __dealloc__(self):
        pass

    def set_allocate(self, allocate):
        self.allocate = allocate

    def get_additional_log_potentials(self):
        return self.thisptr.GetAdditionalLogPotentials()

    def set_additional_log_potentials(self,
                                      vector[double] additional_log_potentials):
        self.thisptr.SetAdditionalLogPotentials(additional_log_potentials)

    def get_degree(self):
        return self.thisptr.Degree()

    def get_link_id(self, int i):
        return self.thisptr.GetLinkId(i)

    def get_variable(self, int i):
        cdef BinaryVariable *variable = self.thisptr.GetVariable(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def solve_map(self, vector[double] variable_log_potentials,
                  vector[double] additional_log_potentials):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveMAP(variable_log_potentials,
                              additional_log_potentials,
                              &posteriors,
                              &additional_posteriors,
                              &value)

        return value, posteriors, additional_posteriors


cdef class PFactorSequence(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequence()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] num_states):
        (<FactorSequence*>self.thisptr).Initialize(num_states)


cdef class PFactorSequenceCompressor(PFactor):
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


cdef class PFactorCompressionBudget(PFactor):
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


cdef class PFactorBinaryTree(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBinaryTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents):
        (<FactorBinaryTree*>self.thisptr).Initialize(parents)


cdef class PFactorBinaryTreeCounts(PFactor):
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


cdef class PFactorGeneralTree(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorGeneralTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] parents, vector[int] num_states):
        (<FactorGeneralTree*>self.thisptr).Initialize(parents, num_states)


cdef class PFactorGeneralTreeCounts(PFactor):
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


cdef class PFactorTree(PFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorTree()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length, list arcs):
        cdef vector[Arc *] arcs_v
        cdef int head, modifier

        cdef tuple arc
        for arc in arcs:
            head = arc[0]
            modifier = arc[1]

            if not 0 <= head < length:
                raise ValueError("Invalid arc: head must be in [0, length)")
            if not 1 <= modifier < length:
                raise ValueError("Invalid arc: modifier must be in [1, length)")
            if not head != modifier:
                raise ValueError("Invalid arc: head cannot be the same as the "
                                 " modifier")
            arcs_v.push_back(new Arc(head, modifier))

        if arcs_v.size() != <Py_ssize_t> self.thisptr.Degree():
            raise ValueError("Number of arcs differs from number of bound "
                             "variables.")
        (<FactorTree*>self.thisptr).Initialize(length, arcs_v)

        for arcp in arcs_v:
            del arcp


cdef int _binary_vars_to_vector(
        list p_vars, vector[BinaryVariable*]& c_vars) except -1:

    cdef PBinaryVariable p_var
    for p_var in p_vars:
        with cython.nonecheck(True):
            c_vars.push_back(p_var.thisptr)
    return 0


cdef int _multi_vars_to_vector(
        list p_vars, vector[MultiVariable*]& c_vars) except -1:

    cdef Py_ssize_t expected = 1
    cdef PMultiVariable p_var
    for p_var in p_vars:
        with cython.nonecheck(True):
            c_vars.push_back(p_var.thisptr)
        expected *= p_var._get_n_states()
    return expected


cdef int _validate_negated(list p_negated, vector[bool]& negated,
                           Py_ssize_t n) except -1:
    try:
        with cython.nonecheck(True):
            (&negated)[0] = p_negated
    except TypeError:
            (&negated)[0] = vector[bool]()

    if negated.size() and negated.size() != n:
        raise ValueError("Expected one negated flag per variable, "
                         "or none at all.")
    return 0


cdef class PFactorGraph:
    """Factor graph instance.

    The main object in AD3, all variables and factors are attached to it.
    """
    cdef FactorGraph *thisptr
    def __cinit__(self):
        self.thisptr = new FactorGraph()

    def __dealloc__(self):
        del self.thisptr

    def set_verbosity(self, int verbosity):
        self.thisptr.SetVerbosity(verbosity)

    def create_binary_variable(self):
        """Creates and returns a new binary variable.

        Returns
        -------

        var, PBinaryVariable
            The created variable.
        """
        cdef BinaryVariable * variable = self.thisptr.CreateBinaryVariable()
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def create_multi_variable(self, int num_states):
        """Creates and returns a new multi-valued variable.

        Parameters
        ----------

        n_states : int
            The number of states the variable may be in.

        Returns
        -------

        var, PMultiVariable
            The created variable.
        """
        cdef MultiVariable * mv = self.thisptr.CreateMultiVariable(num_states)
        pmult = PMultiVariable(allocate=False)
        pmult.thisptr = mv
        return pmult

    def create_factor_logic(self,
                            str factor_type,
                            list p_variables,
                            list negated=None,
                            bool owned_by_graph=True):
        """Create a logic constraint factor and bind it to the variables.

        Parameters
        ----------

        factor_type : string
            One of the following types of logic factor:
            - XOR: allows exactly one variable to be turned on.
            - OR: requires at least one variable to be turned on
            - ATMOSTONE: requires at most one variable to be on
            - IMPLY: if the first n - 1 variables are on, so must the last.
            - XOROUT: XOR with output (see Notes).
            - OROUT: OR with output (see Notes)
            - ANDOUT: AND with output (see Notes)

        p_variables : list of PBinaryVariable objects,
            The bound variables that the factor applies to. For some factors
            the order is meaningful.

        negated : list of bool, optional
            List of boolean flags the same length as ``p_variables``, indicating
            if the output of each variable should be flipped before applying
            the logic factor. By default no variables are flipped.

        owned_by_graph : bool, default: True
            If False, the factor does not get deleted when the factor graph
            gets garbage collected.

        Notes
        -----

        For the factors with output (XOROUT, OROUT, ANDOUT) the last variable
        is the output. For example, in all legal configurations of an XOROUT
        factor over variables (a, b, c), c must be equal to ``a ^ b``.
        """

        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated_
        _binary_vars_to_vector(p_variables, variables)
        _validate_negated(negated, negated_, variables.size())

        if factor_type == 'XOR':
            self.thisptr.CreateFactorXOR(variables, negated_, owned_by_graph)
        elif factor_type == 'XOROUT':
            self.thisptr.CreateFactorXOROUT(variables, negated_, owned_by_graph)
        elif factor_type == 'ATMOSTONE':
            self.thisptr.CreateFactorAtMostOne(variables, negated_,
                                               owned_by_graph)
        elif factor_type == 'OR':
            self.thisptr.CreateFactorOR(variables, negated_, owned_by_graph)
        elif factor_type == 'OROUT':
            self.thisptr.CreateFactorOROUT(variables, negated_, owned_by_graph)
        elif factor_type == 'ANDOUT':
            self.thisptr.CreateFactorANDOUT(variables, negated_, owned_by_graph)
        elif factor_type == 'IMPLY':
            self.thisptr.CreateFactorIMPLY(variables, negated_, owned_by_graph)
        else:
            raise NotImplementedError(
                'Unknown factor type: {}'.format(factor_type))

    def create_factor_pair(self,
                           list p_variables,
                           double edge_log_potential,
                           bool owned_by_graph=True):
        """Create a pair factor between two binary variables.

        Expresses a correlation between both variables being turned on,
        with the specified ``edge_log_potential``. All other configurations
        have a log-potential of 0.

        Parameters
        ----------

        p_variables : list of PBinaryVariable objects,
            The bound variables that the factor applies to. For some factors
            the order is meaningful.

        edge_log_potential : double,
            The score for both variables being turned on simultaneously.

        owned_by_graph : bool, default: True
            If False, the factor does not get deleted when the factor graph
            gets garbage collected.

        """
        cdef vector[BinaryVariable*] variables
        _binary_vars_to_vector(p_variables, variables)
        if variables.size() != 2:
            raise ValueError("Pair factors require exactly two binary "
                             "variables.")
        self.thisptr.CreateFactorPAIR(variables, edge_log_potential,
                                      owned_by_graph)

    def create_factor_budget(self, list p_variables, int budget,
                             list negated=None, bool owned_by_graph=True):
        """Creates and binds a budget factor to the passed binary variables.

        A budget factor limits the maximum amount of variables that can be
        turned on. The variables with highest log-potentials will be selected.

        p_variables : list of PBinaryVariable objects,
            The bound variables that the factor applies to.

        budget : int,
            Maximum number of variables that can be turned on.

        negated : list of bool, optional
            List of boolean flags the same length as ``p_variables``, indicating
            if the output of each variable should be flipped before applying
            the factor. By default no variables are flipped.

        owned_by_graph : bool, default: True
            If False, the factor does not get deleted when the factor graph
            gets garbage collected.

        """
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated_
        _binary_vars_to_vector(p_variables, variables)
        _validate_negated(negated, negated_, variables.size())

        self.thisptr.CreateFactorBUDGET(variables, negated_, budget,
                                        owned_by_graph)

    def create_factor_knapsack(self,
                               list p_variables,
                               vector[double] costs,
                               double budget,
                               list negated=None,
                               bool owned_by_graph=True):
        """Creates and binds a knapsack factor to the passed binary variables.

        A knapsack factor limits the total cost of the active variables. This
        is a weighted version of ``create_factor_budget``.

        p_variables : list of PBinaryVariable objects,
            The bound variables that the factor applies to.

        costs : list of double,
            Costs associated with turning on binary variables. Must have the
            same length as ``p_variables``.

        budget : int,
            Maximum total cost of the variables that can be turned on.

        negated : list of bool, optional
            List of boolean flags the same length as ``p_variables``, indicating
            if the output of each variable should be flipped before applying
            the factor. By default no variables are flipped.

        owned_by_graph : bool, default: True
            If False, the factor does not get deleted when the factor graph
            gets garbage collected.
        """
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated_
        _binary_vars_to_vector(p_variables, variables)
        _validate_negated(negated, negated_, variables.size())

        with cython.nonecheck(True):
            if costs.size() != variables.size():
                raise ValueError("Must provide one cost per variable.")

        self.thisptr.CreateFactorKNAPSACK(variables, negated_, costs, budget,
                                          owned_by_graph)

    def create_factor_dense(self, list p_multi_variables,
                            vector[double] additional_log_potentials,
                            bool owned_by_graph=True):
        """Creates and binds a dense factor to several multi-variables.

        Assigns as potential to each joint assignment of the passed variables.

        p_multi_variables : list of PMultiVariable objects,
            The bound multi-valued variables that the factor applies to.

        additional_log_potentials : list of doubles,
            Log-potentials for each joint configuration, in lexicographic order.
            Required length ``product(len(var) for var in p_multi_variables)``

        owned_by_graph : bool, default: True
            If False, the factor does not get deleted when the factor graph
            gets garbage collected.
        """
        cdef vector[MultiVariable*] multi_variables
        cdef Py_ssize_t n_expected = _multi_vars_to_vector(p_multi_variables,
                                                           multi_variables)

        with cython.nonecheck(True):
            if additional_log_potentials.size() !=  n_expected:
                raise ValueError("Must provide one log-potential per joint "
                                 "state assignment of all the variables.")

        self.thisptr.CreateFactorDense(multi_variables,
                                       additional_log_potentials,
                                       owned_by_graph)

    def declare_factor(self, PFactor p_factor not None,
                       list p_variables, bool owned_by_graph=False):
        """Bind a separately-created factor to variables in the graph.

        Parameters
        ----------

        p_factor : instance of PFactor,
            The instantiated factor to bind to the graph.

        p_variables : list of PBinaryVariable objects,
            The bound variables that the factor applies to.

        owned_by_graph : bool, default: False
            By default, the factor does not get deleted when the factor graph
            gets garbage collected. If True, it will be deleted.

        """
        cdef vector[BinaryVariable*] variables
        _binary_vars_to_vector(p_variables, variables)

        cdef Factor *factor
        if owned_by_graph:
            p_factor.set_allocate(False)
        factor = p_factor.thisptr

        self.thisptr.DeclareFactor(factor, variables, owned_by_graph)

    def fix_multi_variables_without_factors(self):
        """Add one-of-K constraint to unbound multi-variables.

        The well-formedness one-of-K constraint of multi-variables is
        enforced not by the variables but by the dense factors bound to them.
        This function checks for unbound multi-variables and adds XOR logic
        factors on top of them to enforce the constraint. """
        self.thisptr.FixMultiVariablesWithoutFactors()

    def set_eta_psdd(self, double eta):
        self.thisptr.SetEtaPSDD(eta)

    def set_max_iterations_psdd(self, int max_iterations):
        self.thisptr.SetMaxIterationsPSDD(max_iterations)

    def solve_lp_map_psdd(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveLPMAPWithPSDD(&posteriors, &additional_posteriors,
                                        &value)

        return value, posteriors, additional_posteriors

    def set_eta_ad3(self, double eta):
        self.thisptr.SetEtaAD3(eta)

    def adapt_eta_ad3(self, bool adapt):
        self.thisptr.AdaptEtaAD3(adapt)

    def set_max_iterations_ad3(self, int max_iterations):
        self.thisptr.SetMaxIterationsAD3(max_iterations)

    def set_residual_threshold_ad3(self, double threshold):
        self.thisptr.SetResidualThresholdAD3(threshold)

    def solve_lp_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveLPMAPWithAD3(&posteriors,
                                                       &additional_posteriors,
                                                       &value)
        return value, posteriors, additional_posteriors, solver_status

    def solve_exact_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveExactMAPWithAD3(&posteriors,
                                                          &additional_posteriors,
                                                          &value)
        return value, posteriors, additional_posteriors, solver_status

    def get_dual_variables(self):
        return self.thisptr.GetDualVariables()

    def get_local_primal_variables(self):
        return self.thisptr.GetLocalPrimalVariables()

    def get_global_primal_variables(self):
        return self.thisptr.GetGlobalPrimalVariables()

    def solve(self, eta=0.1, adapt=True, max_iter=1000, tol=1e-6,
              ensure_multi_variables=True, verbose=False,
              branch_and_bound=False):
        """Solve the MAP inference problem associated with the factor graph.

        Parameters
        ---------

        eta : float, default: 0.1
            Value of the penalty constant. If adapt_eta is true, this is the
            initial penalty, otherwise every iteration will apply this amount
            of penalty.

        adapt_eta : boolean, default: True
            If true, adapt the penalty constant using the strategy in [2].

        max_iter : int, default: 1000
            Maximum number of iterations to perform.

        tol : double, default: 1e-6
            Theshold for the primal and dual residuals in AD3. The algorithm
            ends early when both residuals are below this threshold.

        ensure_multi_variables : bool, default: True
            The well-formedness one-of-K constraint of multi-variables is
            enforced not by the variables but by the dense factors bound to
            them. This function checks for unbound multi-variables and adds
            XOR logic factors on top of them to enforce the constraint.

        verbose : int, optional
            Degree of verbosity of debugging information to display. By default,
            nothing is printed.

        branch_and_bound : boolean, default: False
            If true, apply a branch-and-bound procedure for obtaining the exact
            MAP (note: this can be slow if the relaxation is "too fractional").

        Returns
        -------

        value : double
            The total score (negative energy) of the solution.

        posteriors : list
            The MAP assignment of each binarized variable in the graph,
            in the order in which they were created. Multi-valued variables
            are represented using a value for each state.  If solution is
            approximate, the values may be fractional.

        additional_posteriors : list
            Additional posteriors for each log-potential in the factors.

        status : string, (integral|fractional|infeasible|unsolved)
            Inference status.
        """

        self.set_verbosity(verbose)
        self.set_eta_ad3(eta)
        self.adapt_eta_ad3(adapt)
        self.set_max_iterations_ad3(max_iter)
        self.set_residual_threshold_ad3(tol)
        if ensure_multi_variables:
            self.fix_multi_variables_without_factors()

        if branch_and_bound:
            result = self.solve_exact_map_ad3()
        else:
            result = self.solve_lp_map_ad3()

        value, marginals, edge_marginals, solver_status = result

        solver_string = ["integral", "fractional", "infeasible", "unsolved"]
        return value, marginals, edge_marginals, solver_string[solver_status]
