#pragma once
#include <thrust/device_vector.h>
#include "triangle.h"
#include "bvh.h"


class mesh {
public:
	__device__ mesh(int num_t, int num_l, int num_m, triangle* triangles,  int* light_indices, material* materials);
	__device__ ~mesh();
    __device__ void set(int num_t, int num_l, int num_m, triangle* triangles, int* light_indices, material* materials);
    //lbvh_node* BVHAccel;
	triangle* m_triangles;
    material* m_materials;
	int* m_light_indices;


    int m_num_triangles;
    int m_num_materials;
    int m_num_lights;

    //int m_num_bvhnode;

    float light_area_sum;

	__device__ bool intersect(ray& ray, float& t, int& id) {
        float temp_t;
        for (int i = 0; i < m_num_triangles; i++)
        {
            if (m_triangles[i].intersect(ray, temp_t))
            {
                if (t > temp_t)
                {
                    t = temp_t;
                    id = i;
                }
            }
        }
        return t < FLT_MAX;
    }


    __device__ void sample_light(vec3& point, int& id,  float& pdf, curandState* rand_state) {
        const float p = light_area_sum * get_rand_float(rand_state);
        float sample_area_sum = 0;
        for(int i = 0; i < m_num_lights; i++)
        {
            sample_area_sum += m_triangles[m_light_indices[i]].m_area;
            if (p <= sample_area_sum)
            {
                float no_use;
                m_triangles[m_light_indices[i]].sample_point(point, no_use, rand_state);
                id = i;
                pdf = 1 / light_area_sum;
                break;
            }
        }
    }
};

__device__
mesh::mesh(int num_t, int num_l, int num_m, triangle* triangles, int* light_indices, material* materials) {
    m_num_triangles = num_t;
    m_num_lights = num_l;
    m_num_materials = num_m;
    m_triangles = triangles;
    m_materials = materials;
    m_light_indices = light_indices;
    light_area_sum = 0;
    for (int i = 0; i < m_num_lights; i++) {
        light_area_sum += m_triangles[light_indices[i]].m_area;
    }
}


__device__
mesh::~mesh() {
    m_triangles = nullptr;
    //m_materials = nullptr;
    m_light_indices = nullptr;
}

__device__ void
mesh::set(int num_t, int num_l, int num_m, triangle* triangles, int* light_indices, material* materials) {
    m_num_triangles = num_t;
    m_num_lights = num_l;
    m_num_materials = num_m;
    m_triangles = triangles;
    m_materials = materials;
    m_light_indices = light_indices;
    light_area_sum = 0;
    for (int i = 0; i < m_num_lights; i++) {
        light_area_sum += m_triangles[light_indices[i]].m_area;
    }
}



