from libcpp.vector cimport vector
from libcpp cimport bool

import pdb

# get the classes from the c++ headers

cdef extern from "ad3/Factor.h" namespace "AD3":
    cdef cppclass BinaryVariable:
        BinaryVariable()
        double GetLogPotential()
        void SetLogPotential(double log_potential)
        int GetId()

    cdef cppclass Factor:
        Factor()
        void SetAdditionalLogPotentials(vector[double] additional_log_potentials)

cdef extern from "ad3/MultiVariable.h" namespace "AD3":
    cdef cppclass MultiVariable:
        int GetNumStates()
        BinaryVariable *GetState(int i)
        double GetLogPotential(int i)
        void SetLogPotential(int i, double log_potential)


cdef extern from "ad3/FactorGraph.h" namespace "AD3":
    cdef cppclass FactorGraph:
        FactorGraph()
        void SetVerbosity(int verbosity)
        void SetEtaAD3(double eta)
        void AdaptEtaAD3(bool adapt)
        void SetMaxIterationsAD3(int max_iterations)
        int SolveLPMAPWithAD3(vector[double]* posteriors,
                              vector[double]* additional_posteriors,
                              double* value)
        int SolveExactMAPWithAD3(vector[double]* posteriors,
                                 vector[double]* additional_posteriors,
                                 double* value)

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
        void DeclareFactor(Factor *factor,
                           vector[BinaryVariable*] variables,
                           bool owned_by_graph)

cdef extern from "examples/dense/FactorSequence.h" namespace "AD3":
    cdef cppclass FactorSequence(Factor):
        FactorSequence()
        void Initialize(vector[int] num_states)

cdef extern from "examples/summarization/FactorSequenceCompressor.h" namespace "AD3":
    cdef cppclass FactorSequenceCompressor(Factor):        
        FactorSequenceCompressor()
        void Initialize(int length, vector[int] left_positions,
                        vector[int] right_positions)

cdef extern from "examples/summarization/FactorBinaryTree.h" namespace "AD3":
    cdef cppclass FactorBinaryTree(Factor):        
        FactorBinaryTree()
        void Initialize(vector[int] parents)

cdef extern from "examples/summarization/FactorGeneralTree.h" namespace "AD3":
    cdef cppclass FactorGeneralTree(Factor):        
        FactorGeneralTree()
        void Initialize(vector[int] parents, vector[int] num_states)

cdef extern from "examples/summarization/FactorGeneralTreeCounts.h" namespace "AD3":
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
        
    def set_additional_log_potentials(self, vector[double] additional_log_potentials):
        self.thisptr.SetAdditionalLogPotentials(additional_log_potentials)
        
        
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


cdef class PFactorGraph:
    cdef FactorGraph *thisptr
    def __cinit__(self):
        self.thisptr = new FactorGraph()

    def __dealloc__(self):
        del self.thisptr

    def set_verbosity(self, int verbosity):
        self.thisptr.SetVerbosity(verbosity)

    def create_binary_variable(self):
        cdef BinaryVariable * variable =  self.thisptr.CreateBinaryVariable()
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

    def set_eta_ad3(self, double eta):
        self.thisptr.SetEtaAD3(eta)

    def adapt_eta_ad3(self, bool adapt):
        self.thisptr.AdaptEtaAD3(adapt)

    def set_max_iterations_ad3(self, int max_iterations):
        self.thisptr.SetMaxIterationsAD3(max_iterations)

    def solve_lp_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveLPMAPWithAD3(&posteriors, &additional_posteriors,
                                       &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors

    def solve_exact_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        self.thisptr.SolveExactMAPWithAD3(&posteriors, &additional_posteriors,
                                          &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors


