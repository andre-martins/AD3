// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.0.
//
// AD3 2.0 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.0 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.0.  If not, see <http://www.gnu.org/licenses/>.

#ifndef FACTOR_BINARY_TREE_COUNTS
#define FACTOR_BINARY_TREE_COUNTS

#include "ad3/GenericFactor.h"
#include "examples/summarization/FactorGeneralTreeCounts.h"
#include <limits>

namespace AD3 {

class FactorBinaryTreeCounts : public FactorGeneralTreeCounts {
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

  double GetCountScore(int count,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    // TODO: allow add hard constraints here.
    return additional_log_potentials[offset_counts_ + count];
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

  double AddCountScore(int count,
                       double weight,
                       vector<double> *variable_posteriors,
                       vector<double> *additional_posteriors) {
    // TODO: allow hard constraints here.
    (*additional_posteriors)[offset_counts_ + count] += weight;
  }


  int GetNumStates(int i) { return 2; }

  int GetCountingState() { return 1; }

 public:
  // Obtain the best configuration.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    // Decode using the Viterbi algorithm.
    int length = GetLength();
    vector<vector<vector<double> > > values(length);
    vector<vector<vector<int> > > path(length);
    vector<vector<vector<int> > > path_bin(length);

    int root = GetRoot();
    RunViterbiForward(variable_log_potentials,
                      additional_log_potentials,
                      root, &values, &path, &path_bin);

    double best_value = MinusInfinity();
    int best_state = -1;
    int best_bin = -1;
    for (int b = 0; b < path[root][0].size(); ++b) {
      int l = path[root][0][b];
      if (l < 0) continue;
      int bin = b;
      if (l == GetCountingState()) --bin;
      double val = values[root][l][bin];
      val += GetCountScore(b, variable_log_potentials,
                           additional_log_potentials);
      //cout << "value[" << b << "] = " << val << endl;
      if (best_state < 0 || val > best_value) {
        best_value = val;
        best_state = l;
        best_bin = b;
      }
    }

    *value = best_value;

    //cout << "best_bin = " << best_bin << endl;
    //cout << "best_value = " << best_value << endl;
    //cout << "Backtracking..." << endl;

    // Path (state sequence) backtracking.
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    assert(sequence->size() == length);
    RunViterbiBacktrack(root, best_state, best_bin,
                        path, path_bin, sequence);

    //cout << "State sequence: ";
    //for (int i = 0; i < num_states_.size(); ++i) {
    //  cout << (*sequence)[i] << "(" << num_states_[i] << ")" << " ";
    //}
    //cout << endl;
  }

  // Compute the score of a given assignment.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int>* sequence =
        static_cast<const vector<int>*>(configuration);
    *value = 0.0;

    int count = 0;
    EvaluateForward(variable_log_potentials,
                    additional_log_potentials,
                    *sequence,
                    GetRoot(),
                    &count,
                    value);

    // Add the score corresponding to the resulting count.
    *value += GetCountScore(count, variable_log_potentials,
                            additional_log_potentials);
  }

  // Given a configuration with a probability (weight), 
  // increment the vectors of variable and additional posteriors.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    const vector<int> *sequence =
        static_cast<const vector<int>*>(configuration);

    int count = 0;
    UpdateMarginalsForward(*sequence,
                           weight,
                           GetRoot(),
                           &count,
                           variable_posteriors,
                           additional_posteriors);

    // Add the score corresponding to the resulting count.
    AddCountScore(count,
                  weight,
                  variable_posteriors,
                  additional_posteriors);
  }

  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<int> *sequence1 =
        static_cast<const vector<int>*>(configuration1);
    const vector<int> *sequence2 =
        static_cast<const vector<int>*>(configuration2);
    assert(sequence1->size() == sequence2->size());
    int count = 0;
    for (int i = 0; i < sequence1->size(); ++i) {
      if ((*sequence1)[i] == (*sequence2)[i] && (*sequence1)[i] == 1) ++count;
    }
    return count;
  }

  // Check if two configurations are the same.
  bool SameConfiguration(
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
  void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    delete sequence;
  }

  Configuration CreateConfiguration() {
    int length = GetLength();
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

    // Set offset_counts right after all the edge additionalvariables.
    offset_counts_ = index;
  }
};

} // namespace AD3

#endif // FACTOR_BINARY_TREE_COUNTS
