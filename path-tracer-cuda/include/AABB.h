#pragma once
#include "ray.h"

struct AABB {
	vec3 min;
	vec3 max;

	__host__ int get_max_extent() {
		vec3 diagonal = max - min;
		if (diagonal.x() > diagonal.y() && diagonal.x() > diagonal.z())
			return 0;
		if (diagonal.y() > diagonal.z())
			return 1;
		return 2;
	}

	__host__ float AABB::get_middle(int axis) {
		return 0.5f * (min[axis] + max[axis]);
	}

	__device__ bool intersect(ray& r) {
		vec3 inv_dir{ 1.0f / r.direction().x(), 1.0f / r.direction().y(), 1.0f / r.direction().z() };
		vec3 t1 = (min - r.origin()) * inv_dir;
		vec3 t2 = (max - r.origin()) * inv_dir;

		float t_enter = fmax(fmax(fmin(t1.x(), t2.x()), fmin(t1.y(), t2.y())), fmin(t1.z(), t2.z()));
		float t_exit = fmin(fmin(fmax(t1.x(), t2.x()), fmax(t1.y(), t2.y())), fmax(t1.z(), t2.z()));
		if (t_enter <= t_exit && t_exit > eps)
		{
			return true;
		}
		return false;
	}
};

inline AABB merge(AABB box, vec3 p) {
	AABB temp;
	temp.min = vec3(fmin(box.min.x(), p.x()), fmin(box.min.y(), p.y()), fmin(box.min.z(), p.z()));
	temp.max = vec3(fmax(box.max.x(), p.x()), fmax(box.max.y(), p.y()), fmax(box.max.z(), p.z()));
	return temp;
}

inline AABB merge(AABB box1, AABB box2) {
	AABB temp;
	temp.min = vec3(fmin(box1.min.x(), box2.min.x()), fmin(box1.min.y(), box2.min.y()), fmin(box1.min.z(), box2.min.z()));
	temp.max = vec3(fmax(box1.max.x(), box2.max.x()), fmax(box1.max.y(), box2.max.y()), fmax(box1.max.z(), box2.max.z()));
	return temp;
}