#pragma once
#include <thrust/device_new.h>

#include "AABB.h"
#include "object.h"
#include "ray.h"
#include "material.h"


struct triangle_param {
	vec3 v0, v1, v2;
	int material_id;
	//tex_coord t0, t1, t2;
};

class triangle{
public:
    size_t m_id;
	vec3 m_v0, m_v1, m_v2; // three vertices
	vec3 m_e1, m_e2; // two edges
	vec3 m_normal; // surface normal
	//tex_coord t0, t1, t2;
	float m_area; // surface area
	material m_material;
    __host__ triangle() = default;
	__host__ __device__ triangle(vec3 v0, vec3 v1, vec3 v2, material m, size_t id);
	//__device__ bool is_light();
    __device__ bool intersect(const ray& ray, float& t);
    __host__ __device__ AABB get_aabb();
	__device__ void sample_point(vec3& sample_point, float& pdf, curandState* local_rand_state) const;
};

__host__ __device__ inline triangle::triangle(vec3 v0, vec3 v1, vec3 v2, material m, size_t id): m_v0(v0), m_v1(v1), m_v2(v2), m_material(m), m_id(id) {
	m_e1 = m_v1 - m_v0;
	m_e2 = m_v2 - m_v0;
	m_area = 0.5f * cross(m_e1, m_e2).length();
	m_normal = normalize(cross(m_e1, m_e2));
}


__device__ inline bool triangle::intersect(const ray& ray, float& t) {
    const vec3 p_vec = cross(ray.direction(), m_e2);
    const float det = dot(m_e1, p_vec);

    // ray parallel to triangle
    if (float_equal(det, 0.0f))
    {
        return false;
    }

    // compute u
    const vec3 t_vec = ray.origin() - m_v0;
    float u = dot(t_vec, p_vec);
    if (u < 0.0f || u > det)
    {
        return false;
    }

    // compute v
    const vec3 q_vec = cross(t_vec, m_e1);
    float v = dot(ray.direction(), q_vec);
    if (v < 0.0f || u + v > det)
    {
        return false;
    }

    const float inv_det = 1.0f / det;
	t = dot(m_e2, q_vec) * inv_det;
    u *= inv_det;
    v *= inv_det;
    if (t < eps)
    {
        return false;
    }
    return true;
}

__host__ __device__ inline AABB triangle::get_aabb() {
    AABB temp;
    return merge(merge(merge(temp, m_v0), m_v1), m_v2);
}

__device__ inline void triangle::sample_point(vec3& sample_point, float& pdf, curandState* local_rand_state) const {
    const float x = std::sqrt(get_rand_float(local_rand_state));
    const float y = get_rand_float(local_rand_state);
    //sample->point = m_v0 * (1 - x) + m_v1 * (1 - y) * x + m_v2 * x * y;
    sample_point = m_v0 * (1 - x) + m_v1 * (1 - y) * x + m_v2 * x * y;
    //sample->normal = this->m_normal;
    //sample.p_m = this->p_m;
    pdf = 1 / m_area;
}




