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

#ifndef FACTOR_COMPRESSION_BUDGET
#define FACTOR_COMPRESSION_BUDGET

#include "ad3/GenericFactor.h"
#include <limits>

namespace AD3 {

class FactorCompressionBudget : public GenericFactor {
 protected:
  double GetNodeScore(int position,
                      int state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    if (state == 0) return 0.0;
    return variable_log_potentials[position];
  }

  // The edge connects node[position-1] to node[position].
  double GetEdgeScore(int position,
                      int previous_state,
                      int state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    int index = index_edges_[position][previous_state][state];
    if (index < 0) {
      // This edge is handled as a variable, rather than an additional
      // variable.
      int index_variable = -index-1;
      //cout << "Using index_variable=" << index_variable << endl;
      return variable_log_potentials[index_variable];
    }
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

  // The edge connects node[position-1] to node[position].
  void AddEdgePosterior(int position,
                        int previous_state,
                        int state,
                        double weight,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
    int index = index_edges_[position][previous_state][state];
    if (index < 0) {
      // This edge is handled as a variable, rather than an additional
      // variable.
      int index_variable = -index-1;
      //cout << "Using index_variable=" << index_variable << endl;
      (*variable_posteriors)[index_variable] += weight;
    } else {
      (*additional_posteriors)[index] += weight;
    }
  }

  int GetNumStates(int position) { return 2; }

  int GetLength() { return length_; }

  int GetBudget() { return budget_; }

  bool CountsForBudget(int position, int state) {
    if (!counts_for_budget_[position]) return false;
    return state == 1;
  }

  bool ExistsBigramVariable(int position) {
    int previous_state = (position >= 0)? 1 : 0;
    int current_state = (position < length_-1)? 1 : 0;
    if (index_edges_[position+1][previous_state][current_state] < 0) {
      return true;
    } else {
      return false;
    }
  }

 public:
  // Obtain the best configuration.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    //cout << "Begin Maximize" << endl;
    // Decode using the Viterbi algorithm.
    int length = GetLength();
    vector<vector<vector<double> > > values(length);
    vector<vector<vector<int> > > path(length);

    // Initialization.
    // Assume budget_ >= 1.
    int num_states = GetNumStates(0);
    values[0].resize(num_states);
    path[0].resize(num_states);
    for (int l = 0; l < num_states; ++l) {
      values[0][l].resize(1);
      path[0][l].resize(1);
      values[0][l][0] =
        GetNodeScore(0, l, variable_log_potentials,
                     additional_log_potentials) +
        GetEdgeScore(0, 0, l, variable_log_potentials,
                     additional_log_potentials);
      path[0][l][0] = -1; // This won't be used.
    }

    //    cout << "chk1" << endl;
    // Recursion.
    for (int i = 0; i < length - 1; ++i) {
      int num_states = GetNumStates(i+1);
      values[i+1].resize(num_states);
      path[i+1].resize(num_states);
      for (int k = 0; k < num_states; ++k) {
        int num_bins = (GetBudget() < i+1)? GetBudget()+1 : i+2;
        values[i+1][k].resize(num_bins);
        path[i+1][k].resize(num_bins);
        for (int b = 0; b < num_bins; ++b) {
          double best_value = -std::numeric_limits<double>::infinity();
          int best = -1;
          for (int l = 0; l < GetNumStates(i); ++l) {
            int bin = b;
            if (CountsForBudget(i, l)) --bin;
            if (bin < 0) continue;
            if (bin >= path[i][l].size()) continue;
            if (i > 0 && path[i][l][bin] < 0) continue;
            //            cout << "Checking " << i << " " << l << " " << bin << " " << values[i][l].size() << endl;
            double val = values[i][l][bin] +
              GetEdgeScore(i+1, l, k, variable_log_potentials,
                           additional_log_potentials);
            if (best < 0 || val > best_value) {
              best_value = val;
              best = l;
            }
          }
          values[i+1][k][b] = best_value +
            GetNodeScore(i+1, k, variable_log_potentials,
                         additional_log_potentials);
          path[i+1][k][b] = best;
          assert(best >= 0);
          //          cout << "V[" << i+1 << ", " << k << ", " << b << "] = "
          //               << values[i+1][k][b] << endl;
          //          cout << "PATH[" << i+1 << ", " << k << ", " << b << "] = "
          //               << path[i+1][k][b] << endl;
        }
      }
    }

    //    cout << "chk2" << endl;
    // Termination.
    double best_value = -std::numeric_limits<double>::infinity();
    int best = -1;
    int best_bin = -1;
    int num_bins = (GetBudget() < length)? GetBudget()+1 : length+1;
    for (int b = 0; b < num_bins; ++b) {
      for (int l = 0; l < GetNumStates(length - 1); ++l) {
        int bin = b;
        if (CountsForBudget(length-1, l)) --bin;
        if (bin < 0) continue;
        if (bin >= path[length-1][l].size()) continue;
        if (length > 1 && path[length-1][l][bin] < 0) continue;
        //        cout << "Checking " << length-1 << " " << l << " " << bin << " " << values[length-1][l].size() << endl;
        double val = values[length-1][l][bin] +
          GetEdgeScore(length, l, 0, variable_log_potentials,
                       additional_log_potentials);
        if (best < 0 || val > best_value) {
          best_value = val;
          best = l;
          best_bin = b;
          //          cout << length-1 << " " << l << " " << b << " " << best_value << " " << val << endl;
        }
      }
    }

    //    cout << best_value << endl;
    //    cout << "chk3" << endl;
    // Path (state sequence) backtracking.
    vector<int> sequence(length);
    sequence[length - 1] = best;
    int b = best_bin;
    for (int i = length - 1; i > 0; --i) {
      //      cout << sequence[i] << " " << path[i].size() << endl;
      //      cout << i << " " << b << " " << path[i][sequence[i]].size() << endl;
      if (CountsForBudget(i, sequence[i])) --b;
      sequence[i - 1] = path[i][sequence[i]][b];
    }

    //    cout << "chk4" << endl;

    vector<int> *selected_nodes = static_cast<vector<int>*>(configuration);
    for (int i = 0; i < length; ++i) {
      if (sequence[i]) selected_nodes->push_back(i);
    }

    *value = best_value;
    //cout << "End Maximize" << endl;
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

    int previous_state = 0;
    for (int i = 0; i < sequence.size(); ++i) {
      int state = sequence[i];
      *value += GetNodeScore(i, state, variable_log_potentials,
                             additional_log_potentials);
      *value += GetEdgeScore(i, previous_state, state,
                             variable_log_potentials,
                             additional_log_potentials);
      previous_state = state;
    }
    *value += GetEdgeScore(sequence.size(), previous_state, 0,
                           variable_log_potentials,
                           additional_log_potentials);
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

    int previous_state = 0;
    for (int i = 0; i < sequence.size(); ++i) {
      int state = sequence[i];
      AddNodePosterior(i, state, weight,
                       variable_posteriors,
                       additional_posteriors);
      AddEdgePosterior(i, previous_state, state, weight,
                       variable_posteriors,
                       additional_posteriors);
      previous_state = state;
    }
    AddEdgePosterior(sequence.size(), previous_state, 0, weight,
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
        // Check if there is a bigram variable that is matched.
        int k = (*selected_nodes1)[i];
        if (ExistsBigramVariable(k-1)) {
          if (k == 0) {
            ++count; // start position.
          } else if (i > 0 && j > 0 &&
                     (*selected_nodes1)[i-1] == k-1 &&
                     (*selected_nodes2)[j-1] == k-1) {
            ++count;
          }
        }
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
    vector<int> *selected_nodes = new vector<int>;
    return static_cast<Configuration>(selected_nodes);
  }

 public:
  // num_states contains the number of states at each position
  // in the sequence.
  // Note: the variables and the the additional log-potentials must be ordered
  // properly.
  // "length" is the length of the sequence. The start and stop positions are not considered here.
  // "budget" is the maximum number of elements that can be active
  // (excluding the ones that do not count to the budget).
  // "count_to_budget" is a boolean vector of length "length" which
  // tells for each element if it counts to the budget.
  // "bigram_positions" is a vector containing the positions where
  // bigrams start (each bigram in this vector will have a score
  // in variable_log_potentials and will correspond to a binary
  // variable).
  void Initialize(int length, int budget,
                  vector<bool> &counts_for_budget,
                  vector<int> &bigram_positions) {
    length_ = length;
    budget_ = budget;
    counts_for_budget_ = counts_for_budget;

    vector<int> bigram_variables(length+1);
    bigram_variables.assign(length+1, -1);
    int index = 0;
    for (int k = 0; k < bigram_positions.size(); ++k) {
      // Note: bigram_position can be -1 for the start position.
      int i = bigram_positions[k] + 1;
      bigram_variables[i] = index;
      ++index;
    }

    // Convention: index_edges[i][j][k] that are negative (say, t < 0)
    // index variables rather than additional variables.
    // The variable index is given by index = -t-1.
    index_edges_.resize(length + 1);
    index = 0;
    for (int i = 0; i <= length; ++i) {
      // If i == 0, the previous state is the start symbol.
      int num_previous_states = (i > 0)? GetNumStates(i - 1) : 1;
      // If i == length-1, the previous state is the final symbol.
      int num_current_states = (i < length)? GetNumStates(i) : 1;
      index_edges_[i].resize(num_previous_states);
      for (int j = 0; j < num_previous_states; ++j) {
        index_edges_[i][j].resize(num_current_states);
        for (int k = 0; k < num_current_states; ++k) {
          if (bigram_variables[i] >= 0 &&
              j == num_previous_states-1 &&
              k == num_current_states-1) {
            // This bigram is handled directly as a variable, and
            // not an additional variable.
            int index_variable = length + bigram_variables[i];
            index_edges_[i][j][k] = -1-index_variable; // Always < 0.
          } else {
            index_edges_[i][j][k] = index;
            ++index;
          }
        }
      }
    }
  }

 private:
  // Budget (maximum number of occurrences of state 0).
  int budget_;
  // Length of the sequence.
  int length_;
  // Tells if each position contributes to the budget.
  vector<bool> counts_for_budget_;
  // At each position, map from edges of states to a global index which
  // matches the index of additional_log_potentials_.
  vector<vector<vector<int> > > index_edges_;
};

} // namespace AD3

#endif // FACTOR_COMPRESSION_BUDGET
