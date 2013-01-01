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
#include <limits>

namespace AD3 {

class FactorGeneralTreeCounts : public GenericFactor {
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

  virtual double GetCountScore(int position,
                               int count,
                               const vector<double> &variable_log_potentials,
                               const vector<double> &additional_log_potentials) {
    if (!IsRoot(position)) return 0.0;
    return variable_log_potentials[offset_counts_ + count];
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

  virtual void AddCountScore(int position,
                             int count,
                             double weight,
                             vector<double> *variable_posteriors,
                             vector<double> *additional_posteriors) {
    if (!IsRoot(position)) return;
    (*variable_posteriors)[offset_counts_ + count] += weight;
  }

  virtual int GetNumStates(int i) { return num_states_[i]; }

  virtual int GetCountingState() { return 0; }

  bool CountsForBudget(int position, int state) {
    if (!counts_for_budget_[position]) return false;
    return state == GetCountingState();
  }

  int GetLength() { return parents_.size(); }

  bool IsLeaf(int i) {
    return children_[i].size() == 0;
  }
  bool IsRoot(int i) { return i == 0; }
  int GetRoot() { return 0; }
  int GetNumChildren(int i) { return children_[i].size(); }
  int GetChild(int i, int t) { return children_[i][t]; }
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

    //cout << "Processing node " << i << "..." << endl;

    // If node is a leaf, seed the values with the node score.
    if (IsLeaf(i)) {
      for (int l = 0; l < num_states; ++l) {
        // Note: we're storing values for the total number of zero-states
        // up to this point, _excluding_ the current node.
        // This way, the number of bins (values to store for each state)
        // equals the number of node descendants including itself.
        (*values)[i][l].resize(1, MinusInfinity());
        (*values)[i][l][0] =
          GetNodeScore(i, l, variable_log_potentials,
                       additional_log_potentials);
        //GetEdgeScore(i, 0, l, variable_log_potentials,
        //               additional_log_potentials);
        //cout << "VALUES[" << i << "][" << l << "][" << 0 << "] = "
        //     << (*values)[i][l][0] << endl;
      }
    } else {
      //cout << "chk 0" << endl;

      // Run Viterbi forward for each child node.
      int num_bins = 0;
      for (int t = 0; t < GetNumChildren(i); ++t) {
        int j = GetChild(i, t);
        RunViterbiForward(variable_log_potentials,
                          additional_log_potentials,
                          j, values, path, path_bin);
        // At this point, we have (*values)[j][l][b] for each
        // child state l and bin b.
        // NOTE: below it could be any other state and not necessarily
        // GetCountingState().
        num_bins += (*values)[j][GetCountingState()].size();
        (*path)[j].resize(num_states);
        (*path_bin)[j].resize(num_states);
      }
      // Increment the number of bins to account for the current node.
      ++num_bins;

      //cout << "Node " << i <<  " has " << num_bins << " descendants" << endl;


      // Initialize values to the node scores.
      for (int k = 0; k < num_states; ++k) {
        (*values)[i][k].resize(num_bins, MinusInfinity());
        for (int b = 0; b < num_bins; ++b) {
          (*values)[i][k][b] =
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
          // NOTE: below it could be any other state and not necessarily
          // GetCountingState().
          int num_bins_child =
            (*values)[j][GetCountingState()].size();
          // We will now take into account the zero-state of the j-th node
          // and save the best label in a higher bin.
          best_labels[t].resize(num_bins_child+1);
          for (int b = 0; b < num_bins_child; ++b) {
            // Seek the best l that yields a count of b.
            // NOTE: if b == 0 exclude the counting state.
            int best = -1;
            double best_value = MinusInfinity();
            for (int l = 0; l < GetNumStates(j); ++l) {
              int bin = b;
              if (CountsForBudget(j, l)) {
                if (b == 0) continue;
                --bin;
              }
              //assert(bin < (*values)[j][l].size());
              // TODO: add + GetCountScore(j, b) below.
              double val = (*values)[j][l][bin] +
                GetCountScore(j, b, variable_log_potentials,
                              additional_log_potentials) +
                GetEdgeScore(j, l, k, variable_log_potentials,
                             additional_log_potentials);
              if (best < 0 || val > best_value) {
                best_value = val;
                best = l;
              }
            }
            // It may happen that best < 0, in which case bin b is impossible
            // for node j.
            //if (best < 0) {
            //  cout << "Bin " << b << " is impossible for node " << j << endl;
            //}
            //assert(best >= 0 || num_bins_child == 1);
            //assert(b < best_labels[t].size());
            //cout << "BEST_LABEL[" << t << "][" << b << "] = " << best << endl; 
            best_labels[t][b] = best;
          }
          if (CountsForBudget(j, GetCountingState())) {
              best_labels[t][num_bins_child] = GetCountingState();
          } else {
              best_labels[t][num_bins_child] = -1;
          }
          //cout << "BEST_LABEL[" << t << "][" << num_bins_child << "] = "
          //     <<  GetCountingState() << endl; 
        }

        // Now, aggregate everything and compute the best values
        // for each bin in the parent node.
        // This is done sequentially, by looking at the children
        // left to right.

        // At this point, num_bins equals the number of descendant nodes
        // (including the current node).
        // best_values will store the best score for getting to state k
        // in the current node, for every possible bin (ignoring the
        // zero-state of the current node).
        // best_bin_sequences will store the best bins for the children
        // nodes that achieve each of the best configurations above.
        vector<double> best_values(num_bins, MinusInfinity());
        vector<vector<int> > best_bin_sequences(num_bins);
        // Total number of bins after merging each child.
        int num_bins_total = 0;
        //best_values[0] = 0.0;
        for (int t = 0; t < GetNumChildren(i); ++t) {
          int j = GetChild(i, t);

          // Make room for path and path_bin.
          (*path)[j][k].resize(num_bins);
          (*path_bin)[j][k].resize(num_bins);

          // NOTE: below it could be any other state and not necessarily
          // GetCountingState().
          int num_bins_child = (*values)[j][GetCountingState()].size();
          vector<double> best_values_partial(1+num_bins_total+num_bins_child,
                                             MinusInfinity());
          vector<vector<int> >
            best_bin_sequences_partial(1+num_bins_total+num_bins_child);

          for (int b = 0; b < 1+num_bins_total+num_bins_child; ++b) {
            int bmin = b - num_bins_total;
            if (bmin < 0) bmin = 0;
            int bmax = b;
            if (bmax > num_bins_child) bmax = num_bins_child;

            // Seek the best (b1,b2) such that b1+b2=b.
            int best_bin = -1;
            double best_value = MinusInfinity();
            for (int b2 = bmin; b2 <= bmax; ++b2) {
              //cout << b2 << " " << bmin << " " << bmax << endl;
              int b1 = b - b2;
              // Check if bin b1 is impossible.
              if (t > 0 && best_bin_sequences[b1].size() == 0) {
                //cout << "bin " << b1 << " is impossible for left siblings of "
                //     << j << " to yield " << b << endl;
                continue;
              }

              assert(b2 < best_labels[t].size());
              int l = best_labels[t][b2];
              if (l < 0) continue;
              int bin2 = b2;
              if (CountsForBudget(j, l)) --bin2;
              //cout << bin2 << " " << l << endl;
              assert(bin2 < (*values)[j][l].size());
              double val = (*values)[j][l][bin2] +
                GetCountScore(j, b2, variable_log_potentials,
                              additional_log_potentials) +
                GetEdgeScore(j, l, k, variable_log_potentials,
                             additional_log_potentials);
              assert(b1 < best_values.size());
              if (t > 0) val += best_values[b1];
              if (best_bin < 0 || val > best_value) {
                best_value = val;
                best_bin = b2;
              }
            }
            best_values_partial[b] = best_value;
            //cout << t << " BEST_VALUES_PARTIAL[" << b << "] = " << best_value << endl;
            if (best_bin < 0) {
              // An empty sequence signals that bin b is impossible.
              //cout << "Bin " << b << " is impossible to be achieved up to node "
              //     << j << endl;
              best_bin_sequences_partial[b].clear();
            } else {
              best_bin_sequences_partial[b] =
                best_bin_sequences[b - best_bin];
              best_bin_sequences_partial[b].push_back(best_bin);
            }
          }

          num_bins_total += num_bins_child;
          //cout << "num_bins_total = " << num_bins_total << endl;
          for (int b = 0; b < num_bins_total + 1; ++b) {
            best_values[b] = best_values_partial[b];
            best_bin_sequences[b] = best_bin_sequences_partial[b];
          }
        }

        //cout << "chk 4" << endl;

        // At this point, num_bins equals the number of descendant nodes
        // (including the current node).
        for (int b = 0; b < num_bins; ++b) {
          //cout << b << endl;
          //cout << "bin=" << bin << endl;
          //cout <<  b << " " << best_values.size() << endl;
          //cout <<  k << " " << (*values)[i].size() << endl;
          //((cout <<  bin << " " << (*values)[i][k].size() << endl;
          assert(b < (*values)[i][k].size());
          if (best_values[b] == MinusInfinity()) {
            //cout << "Node " << i << " and state " << k << " and bin " << b
            //     << " set to -inf" << endl;
          }
          (*values)[i][k][b] += best_values[b];
          //cout << "VALUES[" << i << "][" << k << "][" << b << "] = "
          //     << (*values)[i][k][b] << endl;
          // If the sequence of bins is empty, then this bin is impossible.
          if (best_bin_sequences[b].size() == 0) {
            //cout << "Bin " << b << " is impossible for node " << i
            //     << " and state " << k  << endl;
            for (int t = 0; t < GetNumChildren(i); ++t) {
              int j = GetChild(i, t);
              (*path)[j][k][b] = -1;
              (*path_bin)[j][k][b] = -1;
            }
          } else {
            for (int t = 0; t < GetNumChildren(i); ++t) {
              int j = GetChild(i, t);
              //cout << "best_bin_sequences " << t << " " << best_bin_sequences[b].size() << endl;
              //cout << t << " " << best_bin_sequences[b].size() << endl;
              assert(t < best_bin_sequences[b].size());
              int b2 = best_bin_sequences[b][t];
              assert(b < (*path)[j][k].size());
              //cout << b2 << " " << t << " " << best_labels[t].size() << endl;
              assert(b2 < best_labels[t].size());
              (*path)[j][k][b] = best_labels[t][b2];
              // TODO: maybe below put b2-1 whenever the best label is the
              // zero-label?
              assert(b < (*path_bin)[j][k].size());
              (*path_bin)[j][k][b] = b2;
            }
          }
        }
      }
    }

    if (IsRoot(i)) {
      (*path)[i].resize(1);
      // NOTE: below it could be any other state and not necessarily
      // GetCountingState().
      int num_bins = (*values)[i][GetCountingState()].size();
      //cout << num_bins << " " << num_states_.size() << endl;
      assert(num_bins == GetLength());
      (*path)[i][0].resize(num_bins+1);
      for (int b = 0; b < num_bins; ++b) {
        double best_value;
        int best = -1;
        for (int l = 0; l < num_states; ++l) {
          int bin = b;
          if (CountsForBudget(i, l)) {
            if (b == 0) continue;
            --bin;
          }
          double val = (*values)[i][l][bin];
          //+ GetEdgeScore(i, l, 0, variable_log_potentials,
          //               additional_log_potentials); 
          if (best < 0 || val > best_value) {
            best_value = val;
            best = l;
          }
        }
        (*path)[i][0][b] = best;
      }
      if (CountsForBudget(i, GetCountingState())) {
        (*path)[i][0][num_bins] = GetCountingState();
      } else {
        (*path)[i][0][num_bins] = -1;
      }        
    }
  }

  void RunViterbiBacktrack(int i, int state, int bin,
                           const vector<vector<vector<int> > > &path,
                           const vector<vector<vector<int> > > &path_bin,
                           vector<int> *best_configuration) {
    (*best_configuration)[i] = state;
    if (CountsForBudget(i, state)) --bin;
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
    if (CountsForBudget(i, k)) ++(*count);

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
        int child_count = 0;
        EvaluateForward(variable_log_potentials,
                        additional_log_potentials,
                        configuration,
                        j,
                        &child_count,
                        value);
        *value += GetCountScore(j, child_count,
                                variable_log_potentials,
                                additional_log_potentials);
        *count += child_count;
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
    if (CountsForBudget(i, k)) ++(*count);

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
        int child_count = 0;
        UpdateMarginalsForward(configuration,
                               weight,
                               j,
                               &child_count,
                               variable_posteriors,
                               additional_posteriors);
        AddCountScore(j, child_count, weight,
                      variable_posteriors,
                      additional_posteriors);
        *count += child_count;
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
      if (CountsForBudget(root, l)) --bin;
      double val = values[root][l][bin];
      val += GetCountScore(root, b, variable_log_potentials,
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
    *value += GetCountScore(GetRoot(), count, variable_log_potentials,
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
    AddCountScore(GetRoot(),
                  count,
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
      if (CountsForBudget(i, (*sequence1)[i])) ++b1;
      if (CountsForBudget(i, (*sequence2)[i])) ++b2;
      if ((*sequence1)[i] == (*sequence2)[i]) ++count;
    }
    // TODO: Make sure counts are variables and not additional variables.
    // TODO: This assumes there are no partial counts encoded as variables.
    if (b1 == b2) ++count; 
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
  void Initialize(const vector<int> &parents,
                  const vector<int> &num_states) {
    vector<bool> counts_for_budget(parents.size());
    counts_for_budget.assign(parents.size(), true);
    Initialize(parents, num_states, counts_for_budget);
  }

  void Initialize(const vector<int> &parents,
                  const vector<int> &num_states,
                  vector<bool> &counts_for_budget) {
    int length = parents.size();
    parents_ = parents;
    counts_for_budget_ = counts_for_budget;    
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

 protected:
  // Parent of each node.
  vector<int> parents_;
  // Children of each node.
  vector<vector<int> > children_;
  // Number of states for each position.
  vector<int> num_states_;
  // Tells if each position contributes to the budget.
  vector<bool> counts_for_budget_;
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
