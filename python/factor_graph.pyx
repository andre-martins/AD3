from libcpp.vector cimport vector
from libcpp cimport bool

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
        void StorePrimalDualSequences(bool store)
        void GetPrimalDualSequences(vector[double]* primal_obj_sequence,
                                    vector[double]* dual_obj_sequence)
        void GetNumOracleCallsSequence(vector[int]* num_oracle_calls_sequence)
        void ConvertToBinaryFactorGraph(FactorGraph* binary_factor_graph)
        void SetMaxIterationsMPLP(int max_iterations)
        int SolveLPMAPWithMPLP(vector[double]* posteriors,
                               vector[double]* additional_posteriors,
                               double* value)
        void SetEtaPSDD(double eta)
        void SetMaxIterationsPSDD(int max_iterations)
        int SolveLPMAPWithPSDD(vector[double]* posteriors,
                               vector[double]* additional_posteriors,
                               double* value)
        void SetEtaAD3(double eta)
        void AdaptEtaAD3(bool adapt)
        void SetMaxIterationsAD3(int max_iterations)
        void EnableCachingAD3(bool enable)
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

cdef extern from "../examples/dense/FactorSequence.h" namespace "AD3":
    cdef cppclass FactorSequence(Factor):
        FactorSequence()
        void Initialize(vector[int] num_states)

cdef extern from "../examples/summarization/FactorSequenceCompressor.h" namespace "AD3":
    cdef cppclass FactorSequenceCompressor(Factor):        
        FactorSequenceCompressor()
        void Initialize(int length, vector[int] left_positions,
                        vector[int] right_positions)
                        
cdef extern from "../examples/summarization/FactorCompressionBudget.h" namespace "AD3":
    cdef cppclass FactorCompressionBudget(Factor):        
        FactorCompressionBudget()
        void Initialize(int length, int budget,
                        vector[bool] counts_for_budget,
                        vector[int] bigram_positions)                        

cdef extern from "../examples/summarization/FactorBinaryTree.h" namespace "AD3":
    cdef cppclass FactorBinaryTree(Factor):        
        FactorBinaryTree()
        void Initialize(vector[int] parents)

cdef extern from "../examples/summarization/FactorBinaryTreeCounts.h" namespace "AD3":
    cdef cppclass FactorBinaryTreeCounts(Factor):        
        FactorBinaryTreeCounts()
        void Initialize(vector[int] parents, vector[bool] counts_for_budget)
        void Initialize(vector[int] parents, vector[bool] counts_for_budget,
                        vector[bool] has_count_scores)
        void Initialize(vector[int] parents, vector[bool] counts_for_budget,
                        vector[bool] has_count_scores, int max_num_bins)

cdef extern from "../examples/summarization/FactorGeneralTree.h" namespace "AD3":
    cdef cppclass FactorGeneralTree(Factor):        
        FactorGeneralTree()
        void Initialize(vector[int] parents, vector[int] num_states)

cdef extern from "../examples/summarization/FactorGeneralTreeCounts.h" namespace "AD3":
    cdef cppclass FactorGeneralTreeCounts(Factor):        
        FactorGeneralTreeCounts()
        void Initialize(vector[int] parents, vector[int] num_states)

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
            
    def get_state(self, int i):
        cdef BinaryVariable *variable = self.thisptr.GetState(i)
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def get_log_potential(self, int i):
        return self.thisptr.GetLogPotential(i)

    def set_log_potential(self, int i, double log_potential):
        self.thisptr.SetLogPotential(i, log_potential)


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
        cdef vector[double] additional_log_potentials
        additional_log_potentials = self.thisptr.GetAdditionalLogPotentials()
        p_additional_log_potentials = []
        cdef size_t i
        for i in xrange(additional_log_potentials.size()):
            p_additional_log_potentials.append(additional_log_potentials[i])
        return p_additional_log_potentials
        
    def set_additional_log_potentials(self, vector[double] additional_log_potentials):
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
        self.thisptr.SolveMAP(variable_log_potentials, additional_log_potentials,
                              &posteriors, &additional_posteriors,
                              &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])
            
        return value, p_posteriors, p_additional_posteriors
                
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
        
    def initialize(self, int length, vector[int] left_positions, vector[int] right_positions):
        (<FactorSequenceCompressor*>self.thisptr).Initialize(length, left_positions, right_positions)


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
        (<FactorCompressionBudget*>self.thisptr).Initialize(length, budget, counts_for_budget, bigram_positions)


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
                (<FactorBinaryTreeCounts*>self.thisptr).Initialize(parents,
                                                                   counts_for_budget,
                                                                   has_count_scores,
                                                                   max_num_bins)
            else:
                (<FactorBinaryTreeCounts*>self.thisptr).Initialize(parents,
                                                                   counts_for_budget,
                                                                   has_count_scores)
        else:
            (<FactorBinaryTreeCounts*>self.thisptr).Initialize(parents,
                                                               counts_for_budget)
            
        
        
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
        (<FactorGeneralTreeCounts*>self.thisptr).Initialize(parents, num_states)


cdef class PFactorGraph:
    cdef FactorGraph *thisptr
    cdef bool allocate
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new FactorGraph()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def set_verbosity(self, int verbosity):
        self.thisptr.SetVerbosity(verbosity)

    def store_primal_dual_sequences(self, bool store):
        self.thisptr.StorePrimalDualSequences(store)

    def get_primal_dual_sequences(self):
        cdef vector[double] primal_obj_sequence
        cdef vector[double] dual_obj_sequence
        self.thisptr.GetPrimalDualSequences(&primal_obj_sequence,
                                            &dual_obj_sequence)
        p_primal_obj_sequence, p_dual_obj_sequence = [], []
        cdef size_t i
        for i in range(primal_obj_sequence.size()):
            p_primal_obj_sequence.append(primal_obj_sequence[i])
        for i in range(dual_obj_sequence.size()):
            p_dual_obj_sequence.append(dual_obj_sequence[i])

        return p_primal_obj_sequence, p_dual_obj_sequence

    def get_num_oracle_calls_sequence(self):
        cdef vector[int] num_oracle_calls_sequence
        self.thisptr.GetNumOracleCallsSequence(&num_oracle_calls_sequence)
        p_num_oracle_calls_sequence = []
        cdef size_t i
        for i in range(num_oracle_calls_sequence.size()):
            p_num_oracle_calls_sequence.append(num_oracle_calls_sequence[i])

        return p_num_oracle_calls_sequence

    def convert_to_binary_factor_graph(self):
        cdef FactorGraph * binary_factor_graph = new FactorGraph()
        self.thisptr.ConvertToBinaryFactorGraph(binary_factor_graph)
        p_binary_factor_graph = PFactorGraph(allocate=False)
        p_binary_factor_graph.thisptr = binary_factor_graph
        return p_binary_factor_graph
        
    def create_binary_variable(self):
        cdef BinaryVariable * variable = self.thisptr.CreateBinaryVariable()
        pvariable = PBinaryVariable(allocate=False)
        pvariable.thisptr = variable
        return pvariable

    def create_multi_variable(self, int num_states):
        cdef MultiVariable * mult =  self.thisptr.CreateMultiVariable(num_states)
        pmult = PMultiVariable(allocate=False)
        pmult.thisptr = mult
        return pmult

    def create_factor_logic(self, factor_type, p_variables, p_negated, bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated
        for i, var in enumerate(p_variables):
            variables.push_back((<PBinaryVariable>var).thisptr)
            negated.push_back(p_negated[i])
        if factor_type == 'XOR':
            self.thisptr.CreateFactorXOR(variables, negated, owned_by_graph)
        elif factor_type == 'XOROUT':
            self.thisptr.CreateFactorXOROUT(variables, negated, owned_by_graph)
        elif factor_type == 'ATMOSTONE':
            self.thisptr.CreateFactorAtMostOne(variables, negated, owned_by_graph)
        elif factor_type == 'OR':
            self.thisptr.CreateFactorOR(variables, negated, owned_by_graph)
        elif factor_type == 'OROUT':
            self.thisptr.CreateFactorOROUT(variables, negated, owned_by_graph)
        elif factor_type == 'ANDOUT':
            self.thisptr.CreateFactorANDOUT(variables, negated, owned_by_graph)
        elif factor_type == 'IMPLY':
            self.thisptr.CreateFactorIMPLY(variables, negated, owned_by_graph)
        else:
            print 'Unknown factor type:', factor_type
            raise NotImplementedError

    def create_factor_pair(self, p_variables, double edge_log_potential, bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        for var in p_variables:
            variables.push_back((<PBinaryVariable>var).thisptr)
        self.thisptr.CreateFactorPAIR(variables, edge_log_potential, owned_by_graph)

    def create_factor_budget(self, p_variables, p_negated, int budget, bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated
        for i, var in enumerate(p_variables):
            variables.push_back((<PBinaryVariable>var).thisptr)
            negated.push_back(p_negated[i])
        self.thisptr.CreateFactorBUDGET(variables, negated, budget, owned_by_graph)

    def create_factor_knapsack(self, p_variables, p_negated, p_costs, double budget, bool owned_by_graph=True):
        cdef vector[BinaryVariable*] variables
        cdef vector[bool] negated
        cdef vector[double] costs
        for i, var in enumerate(p_variables):
            variables.push_back((<PBinaryVariable>var).thisptr)
            negated.push_back(p_negated[i])
            costs.push_back(p_costs[i])
        self.thisptr.CreateFactorKNAPSACK(variables, negated, costs, budget, owned_by_graph)

    def create_factor_dense(self,  p_multi_variables, p_additional_log_potentials, bool owned_by_graph=True):
        cdef vector[MultiVariable*] multi_variables
        cdef PMultiVariable blub
        for var in p_multi_variables:
            blub = var
            multi_variables.push_back(<MultiVariable*>blub.thisptr)

        cdef vector[double] additional_log_potentials
        for potential in p_additional_log_potentials:
            additional_log_potentials.push_back(potential)
        self.thisptr.CreateFactorDense(multi_variables, additional_log_potentials, owned_by_graph)
        
    def declare_factor(self, p_factor, p_variables, bool owned_by_graph=False):
        cdef vector[BinaryVariable*] variables
        cdef Factor *factor
        for var in p_variables:
            variables.push_back((<PBinaryVariable>var).thisptr)            
        if owned_by_graph:
            p_factor.set_allocate(False)
        factor = (<PFactor>p_factor).thisptr
        self.thisptr.DeclareFactor(factor, variables, owned_by_graph)

    def fix_multi_variables_without_factors(self):
        self.thisptr.FixMultiVariablesWithoutFactors()

    def set_max_iterations_mplp(self, int max_iterations):
        self.thisptr.SetMaxIterationsMPLP(max_iterations)

    def solve_lp_map(self, algorithm='ad3'):
        if algorithm == 'mplp':
            return self.solve_lp_map_mplp()
        elif algorithm == 'psdd':
            return self.solve_lp_map_psdd()
        elif algorithm == 'ad3':
            return self.solve_lp_map_ad3()
        else:
            print 'Unknown algorithm:', algorithm
            raise NotImplementedError

    def solve_lp_map_mplp(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveLPMAPWithMPLP(&posteriors,
                                                        &additional_posteriors,
                                                        &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def set_eta_psdd(self, double eta):
        self.thisptr.SetEtaPSDD(eta)

    def set_max_iterations_psdd(self, int max_iterations):
        self.thisptr.SetMaxIterationsPSDD(max_iterations)

    def solve_lp_map_psdd(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveLPMAPWithPSDD(&posteriors,
                                                        &additional_posteriors,
                                                        &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def set_eta_ad3(self, double eta):
        self.thisptr.SetEtaAD3(eta)

    def adapt_eta_ad3(self, bool adapt):
        self.thisptr.AdaptEtaAD3(adapt)

    def set_max_iterations_ad3(self, int max_iterations):
        self.thisptr.SetMaxIterationsAD3(max_iterations)

    def enable_caching_ad3(self, bool enable):
        self.thisptr.EnableCachingAD3(enable)

    def solve_lp_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveLPMAPWithAD3(&posteriors,
                                                       &additional_posteriors,
                                                       &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def solve_exact_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveExactMAPWithAD3(&posteriors,
                                                          &additional_posteriors,
                                                          &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors
        
    def get_dual_variables(self):
        cdef vector[double] dual_variables = self.thisptr.GetDualVariables()
        p_dual_variables = []
        for i in xrange(dual_variables.size()):
            p_dual_variables.append(dual_variables[i])
        return p_dual_variables

    def get_local_primal_variables(self):
        cdef vector[double] local_primal_variables = self.thisptr.GetLocalPrimalVariables()
        p_local_primal_variables = []
        for i in xrange(local_primal_variables.size()):
            p_local_primal_variables.append(local_primal_variables[i])
        return p_local_primal_variables

    def get_global_primal_variables(self):
        cdef vector[double] global_primal_variables = self.thisptr.GetGlobalPrimalVariables()
        p_global_primal_variables = []
        for i in xrange(global_primal_variables.size()):
            p_global_primal_variables.append(global_primal_variables[i])
        return p_global_primal_variables

