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

#ifndef FACTOR_GRANDPARENT_HEAD_AUTOMATON
#define FACTOR_GRANDPARENT_HEAD_AUTOMATON

#include "ad3/GenericFactor.h"
#include "FactorHeadAutomaton.h"

namespace AD3 {

class Grandparent {
 public:
 Grandparent(int g, int h, int m) : g_(g), h_(h), m_(m) {}
  ~Grandparent() {}

  int grandparent() { return g_; }
  int head() { return h_; }
  int modifier() { return m_; }

 private:
  int g_;
  int h_;
  int m_;
};

class Grandsibling {
  public:
  Grandsibling(int g, int h, int m, int s) : g_(g), h_(h), m_(m), s_(s) {}
  ~Grandsibling() {}

  int grandparent() { return g_; }
  int head() { return h_; }
  int modifier() { return m_; }
  int sibling() { return s_; }

 private:
  int g_;
  int h_;
  int m_;
  int s_;
};

class FactorGrandparentHeadAutomaton : public GenericFactor {
 public:
  FactorGrandparentHeadAutomaton () {}
  virtual ~FactorGrandparentHeadAutomaton() { ClearActiveSet(); }

  // Compute the score of a given assignment.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    // Decode maximizing over the grandparents and using the Viterbi algorithm
    // as an inner loop.
    int num_grandparents = index_grandparents_.size();
    int best_grandparent = -1;
    int length = length_;
    vector<vector<double> > values(length);
    vector<vector<int> > path(length);
    vector<int> best_path(length);

    // Run Viterbi for each possible grandparent.
    for (int g = 0; g < num_grandparents; ++g) {
      // The start state is m = 0.
      values[0].resize(1);
      values[0][0] = 0.0;
      path[0].resize(1);
      path[0][0] = 0;
      for (int m = 1; m < length; ++m) {
        // m+1 possible states: either keep the previous state (no arc added)
        // or transition to a new state (arc between h and m).
        values[m].resize(m+1);
        path[m].resize(m+1);
        for (int i = 0; i < m; ++i) {
          // In this case, the previous state must also be i.
          values[m][i] = values[m-1][i];
          path[m][i] = i;
        }
        // For the m-th state, the previous state can be anything up to m-1.
        path[m][m] = -1;
        for (int j = 0; j < m; ++j) {
          int index = index_siblings_[j][m];
          double score = values[m-1][j] + additional_log_potentials[index];
          if (use_grandsiblings_) {
            index = index_grandsiblings_[g][j][m];
            if (index >= 0) score += additional_log_potentials[index];
          }
          if (path[m][m] < 0 || score > values[m][m]) {
            values[m][m] = score;
            path[m][m] = j;
          }
        }
        int index = index_grandparents_[g][m];
        values[m][m] += variable_log_potentials[num_grandparents+m-1] +
          additional_log_potentials[index];
      }

      // The end state is m = length.
      int best_last_state = -1;
      double best_score = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < length; ++j) {
        int index = index_siblings_[j][length];
        double score = values[length-1][j] + additional_log_potentials[index];
        if (use_grandsiblings_) {
          index = index_grandsiblings_[g][j][length];
          if (index >= 0) score += additional_log_potentials[index];
        }
        if (best_last_state < 0 || score > best_score) {
          best_score = score;
          best_last_state = j;
        }
      }

      // Add the score of the arc (g-->h).
      best_score += variable_log_potentials[g];

      // Only backtrack if the solution is the best so far.
      if (best_grandparent < 0 || best_score > *value) {
        // This is the best grandparent so far.
        best_grandparent = g;
        *value = best_score;
        best_path[length-1] = best_last_state;

        // Backtrack.
        for (int m = length-1; m > 0; --m) {
          best_path[m-1] = path[m][best_path[m]];
        }
      }
    }

    // Now write the configuration.
    vector<int> *grandparent_modifiers =
      static_cast<vector<int>*>(configuration);
    grandparent_modifiers->push_back(best_grandparent);
    for (int m = 1; m < length; ++m) {
      if (best_path[m] == m) {
        grandparent_modifiers->push_back(m);
      }
    }
  }

  // Compute the score of a given assignment.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int>* grandparent_modifiers =
      static_cast<const vector<int>*>(configuration);
    // Grandparent belong to {0,1,...}
    // Modifiers belong to {1,2,...}
    *value = 0.0;
    int g = (*grandparent_modifiers)[0];
    *value += variable_log_potentials[g];
    int num_grandparents = index_grandparents_.size();
    int m = 0;
    for (int i = 1; i < grandparent_modifiers->size(); ++i) {
      int s = (*grandparent_modifiers)[i];
      *value += variable_log_potentials[num_grandparents+s-1];
      int index = index_siblings_[m][s];
      *value += additional_log_potentials[index];
      if (use_grandsiblings_) {
        index = index_grandsiblings_[g][m][s];
        if (index >= 0) *value += additional_log_potentials[index];
      }
      m = s;
      index = index_grandparents_[g][m];
      *value += additional_log_potentials[index];
    }
    int s = index_siblings_.size();
    int index = index_siblings_[m][s];
    *value += additional_log_potentials[index];
    if (use_grandsiblings_) {
      index = index_grandsiblings_[g][m][s];
      if (index >= 0) *value += additional_log_potentials[index];
    }
  }

  // Given a configuration with a probability (weight),
  // increment the vectors of variable and additional posteriors.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    const vector<int> *grandparent_modifiers =
      static_cast<const vector<int>*>(configuration);
    int g = (*grandparent_modifiers)[0];
    (*variable_posteriors)[g] += weight;
    int num_grandparents = index_grandparents_.size();
    int m = 0;
    for (int i = 1; i < grandparent_modifiers->size(); ++i) {
      int s = (*grandparent_modifiers)[i];
      (*variable_posteriors)[num_grandparents+s-1] += weight;
      int index = index_siblings_[m][s];
      (*additional_posteriors)[index] += weight;
      if (use_grandsiblings_) {
        index = index_grandsiblings_[g][m][s];
        if (index >= 0) (*additional_posteriors)[index] += weight;
      }
      m = s;
      index = index_grandparents_[g][m];
      (*additional_posteriors)[index] += weight;
    }
    int s = index_siblings_.size();
    int index = index_siblings_[m][s];
    (*additional_posteriors)[index] += weight;
    if (use_grandsiblings_) {
      index = index_grandsiblings_[g][m][s];
      if (index >= 0) (*additional_posteriors)[index] += weight;
    }
  }

  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<int> *values1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *values2 = static_cast<const vector<int>*>(configuration2);
    int count = 0;
    if ((*values1)[0] == (*values2)[0]) ++count; // Grandparents matched.
    int j = 1;
    for (int i = 1; i < values1->size(); ++i) {
      for (; j < values2->size(); ++j) {
        if ((*values2)[j] >= (*values1)[i]) break;
      }
      if (j < values2->size() && (*values2)[j] == (*values1)[i]) {
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
    const vector<int> *values1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *values2 = static_cast<const vector<int>*>(configuration2);
    if (values1->size() != values2->size()) return false;
    for (int i = 0; i < values1->size(); ++i) {
      if ((*values1)[i] != (*values2)[i]) return false;
    }
    return true;
  }

  // Delete configuration.
  void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *values = static_cast<vector<int>*>(configuration);
    delete values;
  }

  Configuration CreateConfiguration() {
    // The first element is the index of the grandparent.
    // The remaining elements are the indices of the modifiers.
    vector<int>* grandparent_modifiers = new vector<int>;
    return static_cast<Configuration>(grandparent_modifiers);
  }

 public:
  // Incoming arcs are of the form (g,h) for each g.
  // Outgoing arcs are of the form (h,m) for each m.
  // The variables linked to this factor must be in the same order as
  // the incoming arcs, followed by the outgoing arcs.
  // The incoming arcs must be sorted by grandparent, from smallest to
  // biggest index.
  // The outgoing arcs must be sorted from the closest to the farthest
  // away from the root.
  // Grandparent parts (g, h, m) must include cases where g = m.
  // Factors without incoming arcs must not be created.
  void Initialize(const vector<Arc*> &incoming_arcs,
                  const vector<Arc*> &outgoing_arcs,
                  const vector<Grandparent*> &grandparents,
                  const vector<Sibling*> &siblings) {
    vector<Grandsibling*> grandsiblings;
    Initialize(incoming_arcs, outgoing_arcs, grandparents, siblings,
               grandsiblings);
  }

  void Initialize(const vector<Arc*> &incoming_arcs,
                  const vector<Arc*> &outgoing_arcs,
                  const vector<Grandparent*> &grandparents,
                  const vector<Sibling*> &siblings,
                  const vector<Grandsibling*> &grandsiblings) {
    // length is relative to the head position.
    // E.g. for a right automaton with h=3 and instance_length=10,
    // length = 7. For a left automaton, it would be length = 3.
    use_grandsiblings_ = (grandsiblings.size() > 0);
    int num_grandparents = incoming_arcs.size();
    length_ = outgoing_arcs.size() + 1;
    index_grandparents_.assign(num_grandparents, vector<int>(length_, -1));
    index_siblings_.assign(length_, vector<int>(length_ + 1, -1));
    if (use_grandsiblings_) {
      index_grandsiblings_.assign(num_grandparents,
                                  vector<vector<int> >(length_,
                                                       vector<int>(length_ + 1, -1)));
    }

    // Create a temporary index of modifiers.
    int h = (outgoing_arcs.size() > 0) ? outgoing_arcs[0]->head() : -1;
    int m = (outgoing_arcs.size() > 0) ? outgoing_arcs[0]->modifier() : -1;
    vector<int> index_modifiers(1, 0);
    bool right = (h < m) ? true : false;
    for (int k = 0; k < outgoing_arcs.size(); ++k) {
      int previous_modifier = m;
      m = outgoing_arcs[k]->modifier();

      int position = right ? m - h : h - m;
      index_modifiers.resize(position + 1, -1);
      index_modifiers[position] = k + 1;
    }

    // Construct index of siblings.
    for (int k = 0; k < siblings.size(); ++k) {
      h = siblings[k]->head();
      m = siblings[k]->modifier();
      int s = siblings[k]->sibling();
      // cout << "sibling " << h << " -> " << m << " -> " << s << endl;
      right = (s > h) ? true : false;
      int position_modifier = right ? m - h : h - m;
      int position_sibling = right ? s - h : h - s;
      int index_modifier = index_modifiers[position_modifier];
      int index_sibling = (position_sibling < index_modifiers.size()) ?
        index_modifiers[position_sibling] : length_;

      // Add an offset to save room for the grandparents.
      index_siblings_[index_modifier][index_sibling] =
        grandparents.size() + k;
    }

    // Create a temporary index of grandparents.
    int g = (incoming_arcs.size() > 0) ? incoming_arcs[0]->head() : -1;
    h = (incoming_arcs.size() > 0) ? incoming_arcs[0]->modifier() : -1;
    vector<int> index_incoming;
    for (int k = 0; k < incoming_arcs.size(); ++k) {
      g = incoming_arcs[k]->head();
      // Allow for the case where g is -1 (the head being the root
      // in this case). To handle this, set the position to g+1.
      int position = g + 1;
      index_incoming.resize(position + 1, -1);
      index_incoming[position] = k;
    }

    // Construct index of grandparents.
    for (int k = 0; k < grandparents.size(); ++k) {
      int g = grandparents[k]->grandparent();
      h = grandparents[k]->head();
      m = grandparents[k]->modifier();
      //cout << "grandparent " << g << " -> " << h << " -> " << m << endl;

      right = (m > h) ? true : false;
      int position_modifier = right ? m - h : h - m;
      int position_grandparent = g + 1;
      int index_modifier = index_modifiers[position_modifier];
      int index_grandparent = index_incoming[position_grandparent];
      index_grandparents_[index_grandparent][index_modifier] = k;
    }

    // Construct index of grandsiblings.
    for (int k = 0; k < grandsiblings.size(); ++k) {
      int g = grandsiblings[k]->grandparent();
      h = grandsiblings[k]->head();
      m = grandsiblings[k]->modifier();
      int s = grandsiblings[k]->sibling();

      right = (s > h) ? true : false;
      int position_grandparent = g + 1;
      int position_modifier = right ? m - h : h - m;
      int position_sibling = right ? s - h : h - s;
      int index_grandparent = index_incoming[position_grandparent];
      int index_modifier = index_modifiers[position_modifier];
      int index_sibling = (position_sibling < index_modifiers.size()) ?
        index_modifiers[position_sibling] : length_;

      // Add an offset to save room for the grandparents and siblings.
      index_grandsiblings_[index_grandparent][index_modifier][index_sibling] =
        siblings.size() + grandparents.size() + k;
    }
  }
  // length is relative to the head position. 
  // E.g. for a right automaton with h=3 and instance_length=10,
  // length = 7. For a left automaton, it would be length = 3.
  // (DEPRECATED)
  void Initialize(int length,
                  int num_grandparents,
                  const vector<Sibling*> &siblings,
                  const vector<Grandparent*> &grandparents) {
    length_ = length;
    index_grandparents_.assign(num_grandparents, vector<int>(length, -1));
    index_siblings_.assign(length, vector<int>(length+1, -1));
    for (int k = 0; k < siblings.size(); ++k) {
      int h = siblings[k]->head();
      int m = siblings[k]->modifier();
      int s = siblings[k]->sibling();
      if (s > h) {
        m -= h;
        s -= h;
      } else {
        m = h - m;
        s = h - s;
      }
      index_siblings_[m][s] = grandparents.size() + k;
    }
    for (int k = 0; k < grandparents.size(); ++k) {
      int g = grandparents[k]->grandparent();
      int h = grandparents[k]->head();
      int m = grandparents[k]->modifier();
      if (m > h) {
        m -= h;
      } else {
        m = h - m;
      }
      assert(g >= 0 && g < num_grandparents);
      index_grandparents_[g][m] = k;
    }
  }

 private:
  bool use_grandsiblings_;
  int length_;
  vector<vector<int> > index_siblings_;
  vector<vector<int> > index_grandparents_;
  vector<vector<vector<int> > > index_grandsiblings_;
};

} // namespace AD3

#endif // FACTOR_GRANDPARENT_HEAD_AUTOMATON
