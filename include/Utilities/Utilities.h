// Some functions migrated from rebhuhnc/libraries/Math/easymath.h
#ifndef UTILITIES_H_
#define UTILITIES_H_

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

#include <vector>
#include <math.h>
#include <stdlib.h>
#include <numeric> // used in sortIndices
#include <algorithm> // used in sortIndices

namespace easymath{
// Returns a random number between two values
double rand_interval(double low, double high) ;

// Normalise angles between +/-PI
double pi_2_pi(double) ;

// Sum elements in a vector
double sum(std::vector<double>) ;
} // namespace easymath

template <typename T>
std::vector<T> getMaxIndices(std::vector<T> v);
#include "GetMaxIndices.hpp"

template <typename T>
std::vector<size_t> sortIndices(const std::vector<T> &v);
#include "SortIndices.hpp"

#endif // UTILITIES_H_
