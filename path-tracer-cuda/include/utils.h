#pragma once
#include <curand_kernel.h>
#include <cfloat>
constexpr auto eps = 1e-6f;
constexpr auto pi = 3.14159265358979323846f;

__device__ inline float get_rand_float(curandState* rand_state) {
	return curand_uniform(rand_state);
}

__device__ inline bool float_equal(const float a, const float b) {
	const float dif = (a > b) ? (a - b) : (b - a);
	return dif < eps;
}
