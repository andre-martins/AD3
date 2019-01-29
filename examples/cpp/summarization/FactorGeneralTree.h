// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.1.
//
// AD3 2.1 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.1 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.1.  If not, see <http://www.gnu.org/licenses/>.

#ifndef FACTOR_GENERAL_TREE
#define FACTOR_GENERAL_TREE

#include "ad3/GenericFactor.h"

namespace AD3 {

class FactorGeneralTree : public GenericFactor {
 protected:
    virtual double GetNodeScore(int position,
                                int state,
                                const vector<double> &variable_log_potentials,
                                const vector<double> &additional_log_potentials) {
    return variable_log_potentials[offset_states_[position] + state];
  }

  // The edge connects node[position] to its parent node.
  virtual double GetEdgeScore(int position,
                              int state,
                              int parent_state,
                              const vector<double> &variable_log_potentials,
                              const vector<double> &additional_log_potentials) {
    int index = index_edges_[position][state][parent_state];
    return additional_log_potentials[index];
  }

  virtual void AddNodePosterior(int position,
                                int state,
                                double weight,
                                vector<double> *variable_posteriors,
                                vector<double> *additional_posteriors) {
    (*variable_posteriors)[offset_states_[position] + state] += weight;
  }

  // The edge connects node[position] to its parent node.
  virtual void AddEdgePosterior(int position,
                                int state,
                                int parent_state,
                                double weight,
                                vector<double> *variable_posteriors,
                                vector<double> *additional_posteriors) {
    int index = index_edges_[position][state][parent_state];
    (*additional_posteriors)[index] += weight;
  }

  bool IsLeaf(int i) {
    return children_[i].size() == 0;
  }
  bool IsRoot(int i) { return i == 0; }
  int GetRoot() { return 0; }
  int GetNumChildren(int i) { return children_[i].size(); }
  int GetChild(int i, int t) { return children_[i][t]; }
  virtual int GetNumStates(int i) { return num_states_[i]; }

  void RunViterbiForward(const vector<double> &variable_log_potentials,
                         const vector<double> &additional_log_potentials,
                         int i,
                         vector<vector<double> > *values,
                         vector<vector<int> > *path) {
    int num_states = GetNumStates(i);
    (*values)[i].resize(num_states);

    // If node is a leaf, seed the values with the node score.
    if (IsLeaf(i)) {
      for (int l = 0; l < num_states; ++l) {
        (*values)[i][l] =
          GetNodeScore(i, l, variable_log_potentials,
                       additional_log_potentials);
        //GetEdgeScore(i, 0, l, variable_log_potentials,
        //               additional_log_potentials);
      }
    } else {
      // Initialize values to the node scores.
      for (int k = 0; k < num_states; ++k) {
        (*values)[i][k] = GetNodeScore(i, k, variable_log_potentials,
                                       additional_log_potentials);
      }
      // Increment values with the best transition for each child.
      for (int t = 0; t < GetNumChildren(i); ++t) {
        int j = GetChild(i, t);
        RunViterbiForward(variable_log_potentials,
                          additional_log_potentials,
                          j, values, path);
        (*path)[j].resize(num_states);
        for (int k = 0; k < num_states; ++k) {
          double best_value = -std::numeric_limits<double>::infinity();
          int best = -1;
          for (int l = 0; l < GetNumStates(j); ++l) {
            double val = (*values)[j][l] +
              GetEdgeScore(j, l, k, variable_log_potentials,
                           additional_log_potentials);
            if (best < 0 || val > best_value) {
              best_value = val;
              best = l;
            }
          }
          (*values)[i][k] += best_value;
          (*path)[j][k] = best;
        }
      }
    }

    if (IsRoot(i)) {
      (*path)[i].resize(1);
      double best_value = -std::numeric_limits<double>::infinity();
      int best = -1;
      for (int l = 0; l < num_states; ++l) {
        double val = (*values)[i][l];
          //+ GetEdgeScore(i, l, 0, variable_log_potentials,
          //               additional_log_potentials);
        if (best < 0 || val > best_value) {
          best_value = val;
          best = l;
        }
      }
      (*path)[i][0] = best;
    }
  }

  void RunViterbiBacktrack(int i, int state,
                           const vector<vector<int> > &path,
                           vector<int> *best_configuration) {
    (*best_configuration)[i] = state;
    for (int t = 0; t < GetNumChildren(i); ++t) {
      int j = GetChild(i, t);
      int l = path[j][state];
      RunViterbiBacktrack(j, l, path, best_configuration);
    }
  }

  void EvaluateForward(const vector<double> &variable_log_potentials,
                       const vector<double> &additional_log_potentials,
                       const vector<int> &configuration,
                       int i,
                       double *value) {
    int k = configuration[i];

    if (IsLeaf(i)) {
      *value +=
        GetNodeScore(i, k, variable_log_potentials,
                     additional_log_potentials); //+
      //GetEdgeScore(i, 0, k, variable_log_potentials,
      //               additional_log_potentials);
    } else {
      *value += GetNodeScore(i, k, variable_log_potentials,
                             additional_log_potentials);

      for (int t = 0; t < GetNumChildren(i); ++t) {
        int j = GetChild(i, t);
        int l = configuration[j];
        *value += GetEdgeScore(j, l, k, variable_log_potentials,
                               additional_log_potentials);
        EvaluateForward(variable_log_potentials,
                        additional_log_potentials,
                        configuration,
                        j,
                        value);
      }
    }
  }

  void UpdateMarginalsForward(const vector<int> &configuration,
                              double weight,
                              int i,
                              vector<double> *variable_posteriors,
                              vector<double> *additional_posteriors) {
    int k = configuration[i];

    if (IsLeaf(i)) {
      AddNodePosterior(i, k, weight,
                       variable_posteriors,
                       additional_posteriors);
      //AddEdgePosterior(-1, 0, k, weight,
      //                 variable_posteriors,
      //                 additional_posteriors);
    } else {
      AddNodePosterior(i, k, weight,
                       variable_posteriors,
                       additional_posteriors);
      for (int t = 0; t < GetNumChildren(i); ++t) {
        int j = GetChild(i, t);
        int l = configuration[j];
        AddEdgePosterior(j, l, k, weight,
                         variable_posteriors,
                         additional_posteriors);
        UpdateMarginalsForward(configuration,
                               weight,
                               j,
                               variable_posteriors,
                               additional_posteriors);
      }
    }
  }

 public:
  // Obtain the best configuration.
  virtual void Maximize(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        Configuration &configuration,
                        double *value) {
    // Decode using the Viterbi algorithm.
    int length = num_states_.size();
    vector<vector<double> > values(length);
    vector<vector<int> > path(length);

    int root = GetRoot();
    RunViterbiForward(variable_log_potentials,
                      additional_log_potentials,
                      root, &values, &path);

    int best_state = path[root][0];
    *value = values[root][best_state];

    // Path (state sequence) backtracking.
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    assert(sequence->size() == length);
    RunViterbiBacktrack(root, best_state, path, sequence);
  }

  // Compute the score of a given assignment.
  virtual void Evaluate(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        const Configuration configuration,
                        double *value) {
    const vector<int>* sequence =
        static_cast<const vector<int>*>(configuration);
    *value = 0.0;

    EvaluateForward(variable_log_potentials,
                    additional_log_potentials,
                    *sequence,
                    GetRoot(),
                    value);
  }

  // Given a configuration with a probability (weight),
  // increment the vectors of variable and additional posteriors.
  virtual void UpdateMarginalsFromConfiguration(
      const Configuration &configuration,
      double weight,
      vector<double> *variable_posteriors,
      vector<double> *additional_posteriors) {
    const vector<int> *sequence =
        static_cast<const vector<int>*>(configuration);

    UpdateMarginalsForward(*sequence,
                           weight,
                           GetRoot(),
                           variable_posteriors,
                           additional_posteriors);
  }

  // Count how many common values two configurations have.
  virtual int CountCommonValues(const Configuration &configuration1,
                                const Configuration &configuration2) {
    const vector<int> *sequence1 =
        static_cast<const vector<int>*>(configuration1);
    const vector<int> *sequence2 =
        static_cast<const vector<int>*>(configuration2);
    assert(sequence1->size() == sequence2->size());
    int count = 0;
    for (int i = 0; i < sequence1->size(); ++i) {
      if ((*sequence1)[i] == (*sequence2)[i]) ++count;
    }
    return count;
  }

  // Check if two configurations are the same.
  virtual bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) {
    const vector<int> *sequence1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *sequence2 = static_cast<const vector<int>*>(configuration2);
    assert(sequence1->size() == sequence2->size());
    for (int i = 0; i < sequence1->size(); ++i) {
      if ((*sequence1)[i] != (*sequence2)[i]) return false;
    }
    return true;
  }

  // Delete configuration.
  virtual void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    delete sequence;
  }

  virtual Configuration CreateConfiguration() {
    int length = num_states_.size();
    vector<int>* sequence = new vector<int>(length, -1);
    return static_cast<Configuration>(sequence);
  }

 public:
  // parents contains the parent index of each node.
  // The root must be at position 0, and its parent is -1.
  // num_states contains the number of states at each position
  // in the tree. No start/stop positions are used.
  // Note: the variables and the the additional log-potentials must be ordered
  // properly.
  void Initialize(const vector<int> &parents,
                  const vector<int> &num_states) {
    int length = parents.size();
    parents_ = parents;
    children_.resize(length);
    assert(parents_[0] < 0);
    for (int i = 1; i < length; ++i) {
      assert(parents_[i] >= 0);
      children_[parents_[i]].push_back(i);
    }

    num_states_ = num_states;

    index_edges_.resize(length);
    offset_states_.resize(length);
    int offset = 0;
    for (int i = 0; i < length; ++i) {
      offset_states_[i] = offset;
      offset += num_states_[i];
    }
    int index = 0;
    // Root does not have incoming edges.
    for (int i = 1; i < length; ++i) {
      int p = parents_[i];
      int num_previous_states = num_states_[p];
      int num_current_states = num_states_[i];
      index_edges_[i].resize(num_current_states);
      for (int j = 0; j < num_current_states; ++j) {
        index_edges_[i][j].resize(num_previous_states);
      }
      for (int k = 0; k < num_previous_states; ++k) {
        for (int j = 0; j < num_current_states; ++j) {
          index_edges_[i][j][k] = index;
          ++index;
        }
      }
    }
  }

 protected:
  // Parent of each node.
  vector<int> parents_;
  // Children of each node.
  vector<vector<int> > children_;
  // Number of states for each position.
  vector<int> num_states_;
  // Offset of states for each position.
  vector<int> offset_states_;
  // At each position, map from edges of states to a global index which
  // matches the index of additional_log_potentials_.
  vector<vector<vector<int> > > index_edges_;
};

} // namespace AD3

#endif // FACTOR_GENERAL_TREE
