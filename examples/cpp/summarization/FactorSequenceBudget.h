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

#ifndef FACTOR_SEQUENCE_BUDGET
#define FACTOR_SEQUENCE_BUDGET

#include "ad3/GenericFactor.h"
#include <limits>

namespace AD3 {

class FactorSequenceBudget : public GenericFactor {
 protected:
  double GetNodeScore(int position,
                      int state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    return variable_log_potentials[offset_states_[position] + state];
  }

  // The edge connects node[position-1] to node[position].
  double GetEdgeScore(int position,
                      int previous_state,
                      int state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    int index = index_edges_[position][previous_state][state];
    return additional_log_potentials[index];
  }

  void AddNodePosterior(int position,
                        int state,
                        double weight,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
    (*variable_posteriors)[offset_states_[position] + state] += weight;
  }

  // The edge connects node[position-1] to node[position].
  void AddEdgePosterior(int position,
                        int previous_state,
                        int state,
                        double weight,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
    int index = index_edges_[position][previous_state][state];
    (*additional_posteriors)[index] += weight;
  }

 public:
  // Obtain the best configuration.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    // Decode using the Viterbi algorithm.
    int length = num_states_.size();
    vector<vector<vector<double> > > values(length);
    vector<vector<vector<int> > > path(length);

    // Initialization.
    // Assume budget_ >= 1.
    int num_states = num_states_[0];
    values[0].resize(num_states);
    path[0].resize(num_states);
    for (int l = 0; l < num_states; ++l) {
      if (l == 0) {
        // The state that counts for the budget.
        values[0][l].resize(2);
        path[0][l].resize(2);
      } else {
        values[0][l].resize(1);
        path[0][l].resize(1);
      }
      int bin = 0;
      if (l == 0) ++bin;
      values[0][l][bin] =
        GetNodeScore(0, l, variable_log_potentials,
                     additional_log_potentials) +
        GetEdgeScore(0, 0, l, variable_log_potentials,
                     additional_log_potentials);
      path[0][l][bin] = -1; // This won't be used.
    }

    // Recursion.
    for (int i = 0; i < length - 1; ++i) {
      //cout << i << " " << length - 1 << endl;
      int num_states = num_states_[i+1];
      values[i+1].resize(num_states);
      path[i+1].resize(num_states);
      for (int k = 0; k < num_states; ++k) {
        int num_bins = (budget_ < i+1)? budget_+1 : i+2;
        if (k == 0) {
          // The state that counts for the budget.
          // k == 0 and b = budget_ not allowed.
          if (num_bins == budget_+1) --num_bins;
          values[i+1][k].resize(num_bins+1);
          path[i+1][k].resize(num_bins+1);
        } else {
          values[i+1][k].resize(num_bins);
          path[i+1][k].resize(num_bins);
        }
        for (int b = 0; b < num_bins; ++b) {
          double best_value = -std::numeric_limits<double>::infinity();
          int best = -1;
          for (int l = 0; l < num_states_[i]; ++l) {
            if (l == 0 && b == 0) continue;
            if (i > 0 && path[i][l][b] < 0) continue;
            double val = values[i][l][b] +
              GetEdgeScore(i+1, l, k, variable_log_potentials,
                           additional_log_potentials);
            if (best < 0 || val > best_value) {
              best_value = val;
              best = l;
            }
          }
          int bin = b;
          if (k == 0) ++bin;
          values[i+1][k][bin] = best_value +
            GetNodeScore(i+1, k, variable_log_potentials,
                         additional_log_potentials);
          path[i+1][k][bin] = best;
          //cout << "path[" << i+1 << "][" << k << "][" << bin << "] = " << best << endl;
        }
      }
    }

    // Termination.
    double best_value = -std::numeric_limits<double>::infinity();
    int best = -1;
    int best_bin = -1;
    int num_bins = (budget_ < length)? budget_+1 : length+1;
    for (int b = 0; b < num_bins; ++b) {
      for (int l = 0; l < num_states_[length - 1]; ++l) {
        if (l == 0 && b == 0) continue;
        if (length > 1 && path[length-1][l][b] < 0) continue;
        double val = values[length-1][l][b] +
          GetEdgeScore(length, l, 0, variable_log_potentials,
                       additional_log_potentials);
        if (best < 0 || val > best_value) {
          best_value = val;
          best = l;
          best_bin = b;
        }
      }
    }

    //cout << "here" << endl;

    // Path (state sequence) backtracking.
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    assert(sequence->size() == length);
    (*sequence)[length - 1] = best;
    int b = best_bin;
    for (int i = length - 1; i > 0; --i) {
      //cout << "sequence[" << i << "] = " << (*sequence)[i] << endl;
      //cout << b << endl;
      (*sequence)[i - 1] = path[i][(*sequence)[i]][b];
      if ((*sequence)[i] == 0) --b;
    }

    *value = best_value;
  }

  // Compute the score of a given assignment.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int>* sequence =
        static_cast<const vector<int>*>(configuration);
    *value = 0.0;
    int previous_state = 0;
    for (int i = 0; i < sequence->size(); ++i) {
      int state = (*sequence)[i];
      *value += GetNodeScore(i, state, variable_log_potentials,
                             additional_log_potentials);
      *value += GetEdgeScore(i, previous_state, state,
                             variable_log_potentials,
                             additional_log_potentials);
      previous_state = state;
    }
    *value += GetEdgeScore(sequence->size(), previous_state, 0,
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
    const vector<int> *sequence =
        static_cast<const vector<int>*>(configuration);
    int previous_state = 0;
    for (int i = 0; i < sequence->size(); ++i) {
      int state = (*sequence)[i];
      AddNodePosterior(i, state, weight,
                       variable_posteriors,
                       additional_posteriors);
      AddEdgePosterior(i, previous_state, state, weight,
                       variable_posteriors,
                       additional_posteriors);
      previous_state = state;
    }
    AddEdgePosterior(sequence->size(), previous_state, 0, weight,
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
      if ((*sequence1)[i] == (*sequence2)[i]) ++count;
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
    int length = num_states_.size();
    vector<int>* sequence = new vector<int>(length, -1);
    return static_cast<Configuration>(sequence);
  }

 public:
  // num_states contains the number of states at each position
  // in the sequence. The start and stop positions are not considered here.
  // Note: the variables and the the additional log-potentials must be ordered
  // properly.
  void Initialize(const vector<int> &num_states, int budget) {
    int length = num_states.size();
    budget_ = budget;
    num_states_ = num_states;
    index_edges_.resize(length + 1);
    offset_states_.resize(length);
    int offset = 0;
    for (int i = 0; i < length; ++i) {
      offset_states_[i] = offset;
      offset += num_states_[i];
    }
    int index = 0;
    for (int i = 0; i <= length; ++i) {
      // If i == 0, the previous state is the start symbol.
      int num_previous_states = (i > 0)? num_states_[i - 1] : 1;
      // If i == length-1, the previous state is the final symbol.
      int num_current_states = (i < length)? num_states_[i] : 1;
      index_edges_[i].resize(num_previous_states);
      for (int j = 0; j < num_previous_states; ++j) {
        index_edges_[i][j].resize(num_current_states);
        for (int k = 0; k < num_current_states; ++k) {
          index_edges_[i][j][k] = index;
          ++index;
        }
      }
    }
  }

 private:
  // Budget (maximum number of occurrences of state 0).
  int budget_;
  // Number of states for each position.
  vector<int> num_states_;
  // Offset of states for each position.
  vector<int> offset_states_;
  // At each position, map from edges of states to a global index which
  // matches the index of additional_log_potentials_.
  vector<vector<vector<int> > > index_edges_;
};

} // namespace AD3

#endif // FACTOR_SEQUENCE_BUDGET
