#pragma once
#include "intersection.h"
#include "ray.h"
#include "thrust/device_ptr.h"
#include "path_segment.h"
class object {
public:
	__device__ virtual bool intersect(ray& ray, path_segment& inter) = 0;
};