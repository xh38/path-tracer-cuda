#pragma once
#include "cuda_runtime.h"
#include "ray.h"

struct camera_param {
	vec3 eye;
	vec3 lookat;
	vec3 up;
	float fovy;
	int width;
	int height;
};

class camera {
public:
	__device__ camera(camera_param& cam_param);
	__device__ camera(vec3 eye, vec3 lookat,vec3 up, float fovy, int width, int height);
    __device__ void set(camera_param* cam_param);
	__device__ ray get_ray(int x, int y);
	__device__ ray sample_ray(int x, int y, curandState* rand_state);
	__device__ void init_camera(const vec3 lookat, const vec3 up, const float fovy);

	vec3 m_position;
	vec3 m_upper_left_corner;
	vec3 m_horizontal, m_vertical;
	int m_width, m_height;
};



__device__ inline camera::camera(camera_param& cam_param) {
    m_position = cam_param.eye;
    m_width = cam_param.width;
    m_height = cam_param.height;
    init_camera(cam_param.lookat, cam_param.up, cam_param.fovy);
}

__device__ inline camera::camera(vec3 eye, vec3 lookat, vec3 up, float fovy, int width, int height) {
    m_position = eye;
    m_width = width;
    m_height = height;
    init_camera(lookat, up, fovy);
}

__device__ inline ray camera::get_ray(const int x, const int y) {
    const float s = (static_cast<float>(x) + 0.5f) / static_cast<float>(m_width);
    const float t = (static_cast<float>(y) + 0.5f) / static_cast<float>(m_height);

    return { m_position, m_upper_left_corner + s * m_horizontal - t * m_vertical - m_position };
}

__device__ inline ray camera::sample_ray(const int x, const int y, curandState* rand_state) {
    const float s = (static_cast<float>(x) + get_rand_float(rand_state)) / static_cast<float>(m_width);
    const float t = (static_cast<float>(y) + get_rand_float(rand_state)) / static_cast<float>(m_height);

    return { m_position, m_upper_left_corner + s * m_horizontal - t * m_vertical - m_position };
}

__device__ inline void camera::init_camera(const vec3 lookat, const vec3 up, const float fovy) {
    const float aspect_ratio = static_cast<float>(m_width) / static_cast<float>(m_height);
    const float theta = fovy * pi / 180.0f;
    const float h = tan(theta / 2.0f);
    const float viewport_h = 2.0f * h;
    const float viewport_w = aspect_ratio * viewport_h;

    const vec3 w = normalize(m_position - lookat);
    const vec3 u = normalize(cross(up, w));
    const vec3 v = cross(w, u);

    m_horizontal = viewport_w * u;
    m_vertical = viewport_h * v;

    m_upper_left_corner = m_position - m_horizontal / 2.0f + m_vertical / 2.0f - w;
}

__device__ void camera::set(camera_param* cam_param) {
    m_position = cam_param->eye;
    m_width = cam_param->width;
    m_height = cam_param->height;
    init_camera(cam_param->lookat, cam_param->up, cam_param->fovy);
}

