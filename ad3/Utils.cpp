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

#include "Utils.h"
#include <limits>

namespace AD3 {

int diff_ms(timeval t1, timeval t2) {
  return (((t1.tv_sec - t2.tv_sec) * 1000000) +
	  (t1.tv_usec - t2.tv_usec))/1000;
}

int diff_us(timeval t1, timeval t2) {
  return (((t1.tv_sec - t2.tv_sec) * 1000000000) +
	  (t1.tv_usec - t2.tv_usec));
}

void InsertionSort(pair<double, int> arr[], int length) {
  int i, j;
  pair<double, int> tmp;

  for (i = 1; i < length; i++) {
    j = i;
    while (j > 0 && arr[j - 1].first > arr[j].first) {
      tmp = arr[j];
      arr[j] = arr[j - 1];
      arr[j - 1] = tmp;
      j--;
    }
  }
}

int project_onto_simplex_cached(double* x,
				int d,
				double r, 
				vector<pair<double,int> >& y) {
  int j;
  double s = 0.0;
  double tau;

  // Load x into a reordered y (the reordering is cached).
  if (y.size() != d) {
    y.resize(d);
    for (j = 0; j < d; j++) {
      s += x[j];
      y[j].first = x[j];
      y[j].second = j;
    }
    sort(y.begin(), y.end());
  } else {
    for (j = 0; j < d; j++) {
      s += x[j];
      y[j].first = x[y[j].second];
    }
    // If reordering is cached, use a sorting algorithm 
    // which is fast when the vector is almost sorted.
    InsertionSort(&y[0], d);
  }

  for (j = 0; j < d; j++) {
    tau = (s - r) / ((double) (d - j));
    if (y[j].first > tau) break;
    s -= y[j].first;
  }

  for (j = 0; j < d; j++) {
    if (x[j] < tau) {
      x[j] = 0.0;
    } else {
      x[j] -= tau;
    }
  }

  return 0;
}

int project_onto_simplex(double* x, int d, double r) {
  int j;
  double s = 0.0;
  double tau;
  vector<double> y(d, 0.0);

  for (j = 0; j < d; j++) {
    s += x[j];
    y[j] = x[j];
  }
  sort(y.begin(), y.end());

  for (j = 0; j < d; j++) {
    tau = (s - r) / ((double) (d - j));
    if (y[j] > tau) break;
    s -= y[j];
  }

  for (j = 0; j < d; j++) {
    if (x[j] < tau) {
      x[j] = 0.0;
    } else {
      x[j] -= tau;
    }
  }
   
 return 0;
}

int project_onto_cone_cached(double* x, int d,
			     vector<pair<double,int> >& y) {
  int j;
  double s = 0.0;
  double yav = 0.0;

  if (y.size() != d) {
    y.resize(d);
    for (j = 0; j < d; j++) {
      y[j].first = x[j];
      y[j].second = j;
    }
  } else {
    for (j = 0; j < d; j++) {
      if (y[j].second == d-1 && j != d-1) {
	y[j].second = y[d-1].second;
	y[d-1].second = d-1;
      }
      y[j].first = x[y[j].second];
    }
  }
  InsertionSort(&y[0], d-1);

  for (j = d-1; j >= 0; j--) {
    s += y[j].first;
    yav = s / ((double) (d - j));
    if (j == 0 || yav >= y[j-1].first) break;
  }

  for (; j < d; j++) {
    x[y[j].second] = yav;
  }

  return 0;
}

int project_onto_budget_constraint_cached(double* x,
                                          int d,
                                          double budget, 
                                          vector<pair<double,int> >& y) {
  int j, k, l, level;
  double s = 0.0;
  double tau = 0.0, tightsum;
  double left, right = -std::numeric_limits<double>::infinity();

  // Load x into a reordered y (the reordering is cached).
  if (y.size() != d) {
    y.resize(d);
    for (j = 0; j < d; j++) {
      s -= x[j];
      y[j].first = -x[j];
      y[j].second = j;
    }
    sort(y.begin(), y.end());
  } else {
    for (j = 0; j < d; j++) {
      s -= x[j];
      y[j].first = -x[y[j].second];
    }
    // If reordering is cached, use a sorting algorithm 
    // which is fast when the vector is almost sorted.
    InsertionSort(&y[0], d);
  }

  tightsum = s;
  s += budget;
  
  k = l = level = 0;
  bool found = false;
  double val_a, val_b;
  while (k < d && l < d) {
    if (level != 0) {
      tau = (s - tightsum) / static_cast<double>(level);
    }
    if (k < d) val_a = y[k].first;
    val_b = 1.0 + y[l].first;
    left = right;
    if (k == d || val_b <= val_a) {
      right = val_b;
    } else {
      right = val_a;
    }
    if ((level == 0 && s == tightsum) || (level != 0 && tau <= right)) {
      // Found the right split-point!
      found = true;
      break;
    }
    if (k == d || val_b <= val_a) {
      tightsum += val_b;
      --level;
      ++l;
    } else {
      tightsum -= val_a;
      ++level;
      ++k;
    }
  }

  if (!found) {
    left = right;
    right = std::numeric_limits<double>::infinity();
  }
      
  for (j = 0; j < d; j++) {
    if (-x[j] >= right) {
      x[j] = 0.0;
    } else if (1.0 - x[j] <= left) {
      x[j] = 1.0;
    } else {
      x[j] += tau;
    }
  }

  return 0;
}

int project_onto_budget_constraint(double* x, int d, double budget) {
  int j, k, l, level;
  double s = 0.0;
  double tau = 0.0, tightsum;
  double left, right = -std::numeric_limits<double>::infinity();
  vector<double> y(d, 0.0);

  for (j = 0; j < d; j++) {
    s -= x[j];
    y[j] = -x[j];
  }
  sort(y.begin(), y.end());
  tightsum = s;
  s += budget;
  
  k = l = level = 0;
  bool found = false;
  double val_a, val_b;
  while (k < d && l < d) {
    if (level != 0) {
      tau = (s - tightsum) / static_cast<double>(level);
    }
    if (k < d) val_a = y[k];
    val_b = 1.0 + y[l];
    left = right;
    if (k == d || val_b <= val_a) {
      right = val_b;
    } else {
      right = val_a;
    }
    if ((level == 0 && s == tightsum) || (level != 0 && tau <= right)) {
      // Found the right split-point!
      found = true;
      break;
    }
    if (k == d || val_b <= val_a) {
      tightsum += val_b;
      --level;
      ++l;
    } else {
      tightsum -= val_a;
      ++level;
      ++k;
    }
  }

  if (!found) {
    left = right;
    right = std::numeric_limits<double>::infinity();
  }
      
  for (j = 0; j < d; j++) {
    if (-x[j] >= right) {
      x[j] = 0.0;
    } else if (1.0 - x[j] <= left) {
      x[j] = 1.0;
    } else {
      x[j] += tau;
    }
  }

  return 0;
}

void StringSplit(const string &str,
		 const string &delim,
		 vector<string> *results) {
  size_t cutAt;
  string tmp = str;
  while ((cutAt = tmp.find_first_of(delim)) != tmp.npos) {
    if(cutAt > 0) {
      results->push_back(tmp.substr(0,cutAt));
    }
    tmp = tmp.substr(cutAt+1);
  }
  if(tmp.length() > 0) results->push_back(tmp);
}

void TrimComments(const string &delim, string *line) {
  size_t cutAt = line->find_first_of(delim);
  if (cutAt != line->npos) {
    *line = line->substr(0, cutAt);
  }
}

void TrimLeft(const string &delim, string *line) {
  size_t cutAt = line->find_first_not_of(delim);
  if (cutAt == line->npos) {
    *line == "";
  } else {
    *line = line->substr(cutAt);
  }
}

void TrimRight(const string &delim, string *line) {
  size_t cutAt = line->find_last_not_of(delim);
  if (cutAt == line->npos) {
    *line == "";
  } else {
    *line = line->substr(0, cutAt+1);
  }
}

void Trim(const string &delim, string *line) {
  TrimLeft(delim, line);
  TrimRight(delim, line);
}

} // namespace AD3
