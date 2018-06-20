// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.2.
//
// AD3 2.2 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.2 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.2.  If not, see <http://www.gnu.org/licenses/>.

#ifndef FACTOR_BINARY_TREE
#define FACTOR_BINARY_TREE

#include "ad3/GenericFactor.h"
#include "examples/cpp/summarization/FactorGeneralTree.h"

namespace AD3 {

class FactorBinaryTree : public FactorGeneralTree {
 protected:
  double GetNodeScore(int position,
                      int state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    if (state == 0) return 0.0;
    return variable_log_potentials[position];
  }

  // The edge connects node[position] to its parent node.
  double GetEdgeScore(int position,
                      int state,
                      int parent_state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    int index = index_edges_[position][state][parent_state];
    return additional_log_potentials[index];
  }

  void AddNodePosterior(int position,
                        int state,
                        double weight,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
    if (state == 0) return;
    (*variable_posteriors)[position] += weight;
  }

  // The edge connects node[position] to its parent node.
  void AddEdgePosterior(int position,
                        int state,
                        int parent_state,
                        double weight,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
    int index = index_edges_[position][state][parent_state];
    (*additional_posteriors)[index] += weight;
  }

  int GetNumStates(int i) { return 2; }

  int GetLength() { return parents_.size(); }

 public:
  // Obtain the best configuration.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    // Decode using the Viterbi algorithm.
    int length = GetLength();
    vector<vector<double> > values(length);
    vector<vector<int> > path(length);

    int root = GetRoot();
    RunViterbiForward(variable_log_potentials,
                      additional_log_potentials,
                      root, &values, &path);

    int best_state = path[root][0];
    *value = values[root][best_state];

    // Path (state sequence) backtracking.
    vector<int> sequence(length);
    RunViterbiBacktrack(root, best_state, path, &sequence);
    vector<int> *selected_nodes = static_cast<vector<int>*>(configuration);
    for (int i = 0; i < length; ++i) {
      if (sequence[i]) selected_nodes->push_back(i);
    }
  }

  // Compute the score of a given assignment.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int>* selected_nodes =
        static_cast<const vector<int>*>(configuration);
    *value = 0.0;
    int length = GetLength();
    vector<int> sequence(length, 0);
    for (int k = 0; k < selected_nodes->size(); ++k) {
      int i = (*selected_nodes)[k];
      sequence[i] = 1;
    }

    EvaluateForward(variable_log_potentials,
                    additional_log_potentials,
                    sequence,
                    GetRoot(),
                    value);
  }

  // Given a configuration with a probability (weight),
  // increment the vectors of variable and additional posteriors.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    const vector<int> *selected_nodes =
        static_cast<const vector<int>*>(configuration);

    int length = GetLength();
    vector<int> sequence(length, 0);
    for (int k = 0; k < selected_nodes->size(); ++k) {
      int i = (*selected_nodes)[k];
      sequence[i] = 1;
    }

    UpdateMarginalsForward(sequence,
                           weight,
                           GetRoot(),
                           variable_posteriors,
                           additional_posteriors);
  }

  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<int> *selected_nodes1 =
        static_cast<const vector<int>*>(configuration1);
    const vector<int> *selected_nodes2 =
        static_cast<const vector<int>*>(configuration2);
    int count = 0;
    int j = 0;
    for (int i = 0; i < selected_nodes1->size(); ++i) {
      for (; j < selected_nodes2->size(); ++j) {
        if ((*selected_nodes2)[j] >= (*selected_nodes1)[i]) break;
      }
      if (j < selected_nodes2->size() && (*selected_nodes2)[j] == (*selected_nodes1)[i]) {
        ++count;
        ++j;
      }
    }
    return count;
  }

  // Check if two configurations are the same.
  bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) {
    const vector<int> *selected_nodes1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *selected_nodes2 = static_cast<const vector<int>*>(configuration2);
    if (selected_nodes1->size() != selected_nodes2->size()) return false;
    for (int i = 0; i < selected_nodes1->size(); ++i) {
      if ((*selected_nodes1)[i] != (*selected_nodes2)[i]) return false;
    }
    return true;
  }

  // Delete configuration.
  void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *selected_nodes = static_cast<vector<int>*>(configuration);
    delete selected_nodes;
  }

  Configuration CreateConfiguration() {
    vector<int>* selected_nodes = new vector<int>;
    return static_cast<Configuration>(selected_nodes);
  }

 public:
  // parents contains the parent index of each node.
  // The root must be at position 0, and its parent is -1.
  // num_states contains the number of states at each position
  // in the tree. No start/stop positions are used.
  // Note: the variables and the the additional log-potentials must be ordered
  // properly.
  void Initialize(const vector<int> &parents) {
    int length = parents.size();
    parents_ = parents;
    children_.resize(length);
    assert(parents_[0] < 0);
    for (int i = 1; i < length; ++i) {
      assert(parents_[i] >= 0);
      children_[parents_[i]].push_back(i);
    }

    index_edges_.resize(length);

    int index = 0;
    // Root does not have incoming edges.
    for (int i = 1; i < length; ++i) {
      int p = parents_[i];
      int num_previous_states = 2;
      int num_current_states = 2;
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

};

} // namespace AD3

#endif // FACTOR_GENERAL_TREE
