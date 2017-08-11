from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython
from cpython cimport Py_INCREF
from cython.view cimport array as cvarray
from cython.view cimport array_cwrapper

from base cimport Factor
from base cimport BinaryVariable
from base cimport MultiVariable
from base cimport FactorGraph
from base cimport PBinaryVariable, PMultiVariable, PFactor


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

        cdef Factor* f

        if factor_type == 'XOR':
            f = self.thisptr.CreateFactorXOR(variables, negated_, owned_by_graph)
        elif factor_type == 'XOROUT':
            f = self.thisptr.CreateFactorXOROUT(variables, negated_, owned_by_graph)
        elif factor_type == 'ATMOSTONE':
            f = self.thisptr.CreateFactorAtMostOne(variables, negated_,
                                               owned_by_graph)
        elif factor_type == 'OR':
            f = self.thisptr.CreateFactorOR(variables, negated_, owned_by_graph)
        elif factor_type == 'OROUT':
            f = self.thisptr.CreateFactorOROUT(variables, negated_, owned_by_graph)
        elif factor_type == 'ANDOUT':
            f = self.thisptr.CreateFactorANDOUT(variables, negated_, owned_by_graph)
        elif factor_type == 'IMPLY':
            f = self.thisptr.CreateFactorIMPLY(variables, negated_, owned_by_graph)
        else:
            raise NotImplementedError(
                'Unknown factor type: {}'.format(factor_type))

        cdef PFactor pf = PFactor(allocate=False)
        pf.thisptr = f
        return pf

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
