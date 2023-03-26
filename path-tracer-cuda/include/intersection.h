#pragma once
#include "vector3.h"
#include "material.h"
#include "thrust/device_ptr.h"

class intersection {
public:
	__device__ intersection();
	bool happened;
	float t;
	vec3 point;
	vec3 normal;
	//thrust::device_ptr<material> p_m;
	material* p_m;
	//tex_coord t0, t1, t2;
	//vec3 v0, v1, v2;
	//float u, v;
};

__device__ inline intersection::intersection() {
	happened = false;
	t = 0.0f;
	point = vec3(0, 0, 0);
	normal = point;
	p_m = nullptr;
}

