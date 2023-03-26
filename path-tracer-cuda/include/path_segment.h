#pragma once
#include "vector3.h"
#include "material.h"
struct path_segment {
	// tracing state
	bool end;
	ray r;

	// hit record
	int hit_idx;
	float t;

	// color calculated
	vec3 color;
	vec3 color_all;
	vec3 color_indirect;
	
	//curandState* rand_state;
};
