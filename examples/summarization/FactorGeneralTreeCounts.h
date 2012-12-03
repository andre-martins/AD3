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

#ifndef FACTOR_GENERAL_TREE_COUNTS
#define FACTOR_GENERAL_TREE_COUNTS

#include "ad3/GenericFactor.h"

namespace AD3 {

class FactorGeneralTreeCounts : public GenericFactor {
 protected:
  double GetNodeScore(int position,
                      int state,
                      const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials) {
    return variable_log_potentials[offset_states_[position] + state];
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
    return variable_log_potentials[offset_counts_ + count];
  }

  void AddNodePosterior(int position,
                        int state,
                        double weight,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
    (*variable_posteriors)[offset_states_[position] + state] += weight;
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
    (*variable_posteriors)[offset_counts_ + count] += weight;
  }

  bool IsLeaf(int i) {
    return children_[i].size() == 0;
  }
  bool IsRoot(int i) { return i == 0; }
  int GetRoot() { return 0; }
  int GetNumChildren(int i) { return children_[i].size(); }
  int GetChild(int i, int t) { return children_[i][t]; }
  int GetNumStates(int i) { return num_states_[i]; }
  int GetCountingState() { return 0; }
  double MinusInfinity() {
    return -std::numeric_limits<double>::infinity();
  }

  void RunViterbiForward(const vector<double> &variable_log_potentials,
                         const vector<double> &additional_log_potentials,
                         int i,
                         vector<vector<vector<double> > > *values,
                         vector<vector<vector<int> > > *path,
                         vector<vector<vector<int> > > *path_bin) {
    int num_states = GetNumStates(i);
    (*values)[i].resize(num_states);

    // If node is a leaf, seed the values with the node score.
    if (IsLeaf(i)) {
      for (int l = 0; l < num_states; ++l) {
        int bin = 0;
        // The state that we are counting.
        if (l == GetCountingState()) ++bin;
        (*values)[i][l].resize(bin+1, MinusInfinity());
        (*values)[i][l][bin] =
          GetNodeScore(i, l, variable_log_potentials,
                       additional_log_potentials);
        //GetEdgeScore(i, 0, l, variable_log_potentials,
        //               additional_log_potentials); 
      }
    } else {
      // Run Viterbi forward for each child node.
      int num_bins = 0;
      for (int t = 0; t < GetNumChildren(i); ++t) {
        int j = GetChild(i, t);
        RunViterbiForward(variable_log_potentials,
                          additional_log_potentials,
                          j, values, path, path_bin);
        // At this point, we have (*values)[j][l][b] for each
        // child state l and bin b.
        num_bins += (*values)[j][GetCountingState()].size();
        (*path)[j].resize(num_states);
        (*path_bin)[j].resize(num_states);
      }

      // Initialize values to the node scores.
      for (int k = 0; k < num_states; ++k) {
        int num_bins_state = num_bins;
        if (k == GetCountingState()) ++num_bins_state;
        (*values)[i][k].resize(num_bins_state, MinusInfinity());
        for (int b = 0; b < num_bins; ++b) {
          int bin = b;
          if (k == GetCountingState()) ++bin;
          (*values)[i][k][bin] =
            GetNodeScore(i, k, variable_log_potentials,
                         additional_log_potentials);
        }
      }

      for (int k = 0; k < num_states; ++k) {
        // Incorporate the transitions. For each child node 
        // and for each bin (in the child node) get the best label 
        // and corresponding value for that bin.
        // This can be done for each child node independently.
        vector<vector<int> > best_labels(GetNumChildren(i));
        for (int t = 0; t < GetNumChildren(i); ++t) {
          int j = GetChild(i, t);
          int num_bins_child = 
            (*values)[j][GetCountingState()].size();
          best_labels[t].resize(num_bins_child);

          for (int b = 0; b < num_bins_child; ++b) {
            // Seek the best l that yields a count of b.
            int best = -1;
            double best_value = MinusInfinity();
            for (int l = 0; l < GetNumStates(j); ++l) {
              double val = (*values)[j][l][b] + 
                GetEdgeScore(j, l, k, variable_log_potentials,
                             additional_log_potentials);
              if (best < 0 || val > best_value) {
                best_value = val;
                best = l;
              }
            }
            best_labels[t][b] = best;
          }
        }

        // Now, aggregate everything and compute the best values
        // for each bin in the parent node.
        // This is done sequentially, by looking at the children
        // left to right.
        vector<double> best_values(num_bins, MinusInfinity());
        vector<vector<int> > best_bin_sequences(num_bins);
        int num_bins_total = 1;
        best_values[0] = 0.0;
        for (int t = 0; t < GetNumChildren(i); ++t) {
          int j = GetChild(i, t);

          // Make room for path and path_bin.
          int num_bins_state = num_bins;
          if (k == GetCountingState()) ++num_bins_state;
          (*path)[j][k].resize(num_bins_state);
          (*path_bin)[j][k].resize(num_bins_state);

          int num_bins_child =
            (*values)[j][GetCountingState()].size();
          vector<double> best_values_partial(num_bins_total +
                                             num_bins_child - 1,
                                             MinusInfinity());
          vector<vector<int> >
            best_bin_sequences_partial(num_bins_total +
                                       num_bins_child - 1);

          for (int b = 0;
               b < num_bins_total + num_bins_child - 1; 
               ++b) {
            int bmin = b - num_bins_total + 1;
            if (bmin < 0) bmin = 0;
            int bmax = b;
            if (bmax >= num_bins_child) bmax = num_bins_child-1;

            // Seek the best (b1,b2) such that b1+b2=b.
            int best_bin = -1;
            int best = -1;
            double best_value = MinusInfinity();
            for (int b2 = bmin; b2 <= bmax; ++b2) {
              int b1 = b - b2;
              int l = best_labels[t][b2];
              double val = (*values)[j][b2][l] + 
                GetEdgeScore(j, l, k, variable_log_potentials,
                             additional_log_potentials);
              val += best_values[b1];
              if (best < 0 || val > best_value) {
                best_value = val;
                best_bin = b2;
              }
            }
            best_values_partial[b] = best_value;
            best_bin_sequences_partial[b] =
              best_bin_sequences[b - best_bin];
            best_bin_sequences_partial[b].push_back(best_bin);
          }

          num_bins_total += num_bins_child - 1;
          for (int b = 0; b < num_bins_total; ++b) {
            best_values[b] = best_values_partial[b];
            best_bin_sequences[b] = best_bin_sequences_partial[b];
          }
        }

        for (int b = 0; b < num_bins; ++b) {
          int bin = b;
          if (k == GetCountingState()) ++bin;
          (*values)[i][k][bin] += best_values[b];
          for (int t = 0; t < GetNumChildren(i); ++t) {
            int j = GetChild(i, t);
            int b2 = best_bin_sequences[b][t];
            (*path)[j][k][bin] = best_labels[t][b2];
            (*path_bin)[j][k][bin] = b2;
          }
        }
      }
    }

    if (IsRoot(i)) {
      (*path)[i].resize(1);
      int num_bins = (*values)[i][GetCountingState()].size();
      assert(num_bins == num_states_.size()+1);
      for (int b = 0; b < num_bins; ++b) {
        double best_value;
        int best = -1;
        for (int l = 0; l < num_states; ++l) {
          // TODO: don't try l == 0 and b == 0, or
          // l != 0 and b == num_bins-1.
          double val = (*values)[i][l][b]; 
          //+ GetEdgeScore(i, l, 0, variable_log_potentials,
          //               additional_log_potentials); 
          if (best < 0 || val > best_value) {
            best_value = val;
            best = l;
          }
        }
        (*path)[i][0][b] = best;
      }
    }
  }

  void RunViterbiBacktrack(int i, int state, int bin,
                           const vector<vector<vector<int> > > &path,
                           const vector<vector<vector<int> > > &path_bin,
                           vector<int> *best_configuration) {
    (*best_configuration)[i] = state;
    for (int t = 0; t < GetNumChildren(i); ++t) {
      int j = GetChild(i, t);
      int l = path[j][state][bin];
      int b = path_bin[j][state][bin];
      RunViterbiBacktrack(j, l, b, path, path_bin, best_configuration);
    }
  }

  void EvaluateForward(const vector<double> &variable_log_potentials,
                       const vector<double> &additional_log_potentials,
                       const vector<int> &configuration,
                       int i,
                       int *count,
                       double *value) {
    int num_states = GetNumStates(i);
    int k = configuration[i];
    if (k == GetCountingState()) ++(*count);

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
                        count,
                        value);
      }
    }
  }

  void UpdateMarginalsForward(const vector<int> &configuration,
                              double weight,
                              int i,
                              int *count,
                              vector<double> *variable_posteriors,
                              vector<double> *additional_posteriors) {
    int num_states = GetNumStates(i);
    int k = configuration[i];
    if (k == GetCountingState()) ++(*count);

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
                               count,
                               variable_posteriors,
                               additional_posteriors);
      }
    }
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
    vector<vector<vector<int> > > path_bin(length);

    int root = GetRoot();
    RunViterbiForward(variable_log_potentials,
                      additional_log_potentials,
                      root, &values, &path, &path_bin);

    double best_value;
    int best_state = -1;
    int best_bin = -1;
    for (int b = 0; b < path[root][0].size(); ++b) {
      int l = path[root][0][b];
      double val = values[root][best_state][b];
      val += GetCountScore(b, variable_log_potentials,
                           additional_log_potentials);
      if (best_state < 0 || val > best_value) {
        best_value = val;
        best_state = l;
      }
    }
    
    *value = best_value;

    // Path (state sequence) backtracking.
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    assert(sequence->size() == length);
    RunViterbiBacktrack(root, best_state, best_bin, 
                        path, path_bin, sequence);
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
    int b1 = 0;
    int b2 = 0;
    for (int i = 0; i < sequence1->size(); ++i) {
      if ((*sequence1)[i] == GetCountingState()) ++b1;
      if ((*sequence2)[i] == GetCountingState()) ++b2;
      if ((*sequence1)[i] == (*sequence2)[i]) ++count;
    }
    if (b1 == b2) ++count; // TODO: Make sure counts are variables and not additional variables.
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
    
    // Set offset_counts right after all the state variables.
    offset_counts_ = offset;

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

 private:
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
  // Offset of counts.
  int offset_counts_;
};

} // namespace AD3

#endif // FACTOR_GENERAL_TREE_COUNTS
