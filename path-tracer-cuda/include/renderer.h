#pragma once
#include "mesh.h"
#include "camera.h"
#include <cuda_runtime.h>
//__global__ void render(vec3* frame_buffer, camera* pCamera,  mesh* pMesh, curandState* rand_state, path_segment* segments, int spp) {
//	const int x = blockIdx.x * blockDim.x + threadIdx.x;
//	const int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (x >= pCamera->m_width || y >= pCamera->m_height)
//	{
//		return;
//	}
//	const int pixel_index = y * pCamera->m_width + x;
//
//	curandState local_rand_state = rand_state[pixel_index];
//	path_segment local_render_state = segments[pixel_index];
//
//	ray r = pCamera->sample_ray(x, y, &local_rand_state);
//	frame_buffer[pixel_index] += trace_ray(r, pMesh, local_render_state, &local_rand_state) / static_cast<float>(spp);
//		
//	
//	rand_state[pixel_index] = local_rand_state;
//}


//__global__ void path_trace(path_segment* segments, mesh& p_mesh, curandState* rand_state, int width, int height) {
//	const int x = blockIdx.x * blockDim.x + threadIdx.x;
//	const int y = blockIdx.y * blockDim.y + threadIdx.y;
//	path_segment& local_segment = segments[y * width + x];
//	if (x >= width || y >= height)
//	{
//		return;
//	}
//	ray& r = local_segment.r;
//	size_t intersection_id = -1;
//	float t = FLT_MAX;
//	if(p_mesh.intersect(r, t, intersection_id))
//	{
//		vec3 hit_point = r(t);
//		const triangle hit_obj = p_mesh.m_triangles[intersection_id];
//		vec3 hit_normal = hit_obj.m_normal;
//		const material hit_material = hit_obj.m_material;
//		if(hit_material.is_light())
//		{
//			local_segment.color = hit_material.get_radiance();
//		}
//
//		float pdf_light;
//		vec3 light_point, light_normal, radinace;
//		p_mesh.sample_light(light_point, light_normal, radinace, pdf_light, rand_state);
//
//	}
//}
__device__ vec3 to_world(const vec3& local, const vec3& normal) {
	vec3 local_y;
	if (std::fabs(normal.x()) > std::fabs(normal.y()))
	{
		const float inv_length = 1.0f / sqrt(normal.x() * normal.x() + normal.z() * normal.z());
		local_y = vec3{ normal.z() * inv_length, 0.0, -normal.x() * inv_length };
	}
	else
	{
		const float inv_length = 1.0f / sqrt(normal.y() * normal.y() + normal.z() * normal.z());
		local_y = vec3{ 0.0, normal.z() * inv_length,  -normal.y() * inv_length };
	}
	const vec3 local_x = cross(local_y, normal);
	return local.x() * local_x + local.y() * local_y + local.z() * normal;
}

__device__ vec3 cosine_weighted_sample(vec3 in_dir, vec3 normal, float& pdf, curandState* rand_state) {
	float x1 = get_rand_float(rand_state);
	float x2 = get_rand_float(rand_state);
	float phi = 2 * pi * x2;
	float z = sqrt(x1);
	float sin_theta = sqrt(1.0 - z * z);
	vec3 local_ray_dir{ sin_theta * std::cos(phi), sin_theta * std::sin(phi), z };
	pdf = z / pi;
	return to_world(local_ray_dir, normal);
}



__global__ void generate_ray(path_segment* dev_paths, camera* dev_camera, curandState* rand_states) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x >= dev_camera->m_width || y >= dev_camera->m_height)
	{
		return;
	}
	size_t idx = x + y * dev_camera->m_width;
	curandState local_rand_state = rand_states[idx];
	//dev_paths[idx].pixel_id = idx;
	dev_paths[idx].r = dev_camera->sample_ray(x, y, &local_rand_state);
	dev_paths[idx].end = false;
	dev_paths[idx].color = vec3(1.0, 1.0, 1.0);
	dev_paths[idx].color_indirect = vec3(1.0, 1.0, 1.0);
	dev_paths[idx].color_all = vec3(0, 0, 0);
	
	//dev_paths[idx].rand_state = &local_rand_state;
}

__global__ void compute_intersection(path_segment* dev_paths, mesh* dev_mesh, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
	int path_idx = x + y * width;
	path_segment& local_path = dev_paths[path_idx];
	local_path.t = FLT_MAX;
	local_path.hit_idx = -1;
	dev_mesh->intersect(local_path.r, local_path.t, local_path.hit_idx);
}

__global__ void shade_path_segment(path_segment* dev_paths, mesh* dev_mesh, int width, int height, curandState* rand_state, int depth) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
	{
		return;
	}
	int path_idx = x + y * width;
	path_segment& local_path = dev_paths[path_idx];
	if (local_path.end)
	{
		return;
	}

	if (local_path.hit_idx == -1)
	{
		local_path.color = vec3(0, 0, 0);
		local_path.end = true;
	}

	triangle hit_triangle = dev_mesh->m_triangles[local_path.hit_idx];
	material hit_material = dev_mesh->m_materials[hit_triangle.m_material_id];

	if (hit_material.is_light())
	{
		// only direct hit to the light is allowed
		if (depth == 0)
		{
			local_path.color_all = hit_material.get_radiance();
			local_path.end = true;
		} else
		{
			local_path.end = true;
		}
	} else {
		ray r = local_path.r;
		vec3 hit_normal = hit_triangle.m_normal;
		vec3 hit_point = r(local_path.t) + eps * hit_normal;

		vec3 light_point;
		float light_pdf;
		int light_id;
		dev_mesh->sample_light(light_point, light_id, light_pdf, &rand_state[path_idx]);
		vec3 light_normal = dev_mesh->m_triangles[dev_mesh->m_light_indices[light_id]].m_normal;
		vec3 radiance = dev_mesh->m_materials[dev_mesh->m_triangles[dev_mesh->m_light_indices[light_id]].m_material_id].get_radiance();

		vec3 shadow_ray_vec = light_point - hit_point;
		vec3 shaow_ray_dir = normalize(shadow_ray_vec);
		ray shadow_ray = ray(hit_point, shaow_ray_dir);

		float temp_t = FLT_MAX;
		int hit_id = -1;
		dev_mesh->intersect(shadow_ray, temp_t, hit_id);
		if(hit_id == -1 || hit_id == light_id)
		{
			local_path.color_all += local_path.color_indirect * radiance * hit_material.lambertian_brdf(hit_normal, shaow_ray_dir) * dot(-shaow_ray_dir, light_normal) / light_pdf / shadow_ray_vec.length_squared();
		}

		if(get_rand_float(&rand_state[path_idx]) > 0.8)
		{
			local_path.end = true;
		} else
		{
			float pdf;
			vec3 new_dir = cosine_weighted_sample(local_path.r.m_direction, hit_triangle.m_normal, pdf, &rand_state[path_idx]);
			local_path.color_indirect *= hit_material.lambertian_brdf(hit_normal, new_dir) * dot(new_dir, hit_normal) / pdf / 0.8f;
			local_path.r = ray(hit_point, new_dir);
		}
	}
}

__global__ void fill_frame_buffer(vec3* frame_buffer, path_segment* dev_paths, int width, int height, int spp) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}
	int path_idx = x + y * width;
	frame_buffer[path_idx] += dev_paths[path_idx].color_all / static_cast<float>(spp);
}
