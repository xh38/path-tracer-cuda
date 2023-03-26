#pragma once
#include "vector3.h"

class ray
{
public:
    __device__ ray() = default;
    __device__ ray(const vec3& a, const vec3& b) { m_origin = a; m_direction = b; }
    __device__ vec3 origin() const { return m_origin; }
    __device__ vec3 direction() const { return m_direction; }
    //__device__ vec3 point_at_parameter(const float t) const { return m_origin + t * m_direction; }
    __device__ vec3 operator()(const float t) const { return m_origin + m_direction * t; }

    vec3 m_origin;
    vec3 m_direction;
};