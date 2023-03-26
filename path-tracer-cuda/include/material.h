#pragma once
#include <string>

#include "vector3.h"

struct material_param {
	vec3 kd, ks, tr;
	float ns, ni;
	bool hasEmit;
	vec3 radiance;
};


class material {
public:
	__host__ material() = default;
	__host__ __device__ material(material_param& param)
	: kd(param.kd), ks(param.ks), tr(param.tr),
	  ns(param.ns), ni(param.ni), hasEmit(param.hasEmit), radiance(param.radiance) {}
	
	vec3 kd, ks, tr;
	float ns, ni;
	bool hasEmit;
	vec3 radiance;
	//bool hasTexture;
	//Texture texture;

	__device__ static vec3 reflect(const vec3& in, const vec3& normal) {
		return in - 2 * dot(in, normal) * normal;
	}

	__device__ bool is_light() const { return hasEmit; }

	__device__ vec3 get_radiance() const {
		return radiance;
	}

	//__device__ vec3 uniform_sample(vec3 in_dir, vec3 normal, float& pdf);
	__device__ vec3 cosine_weighted_sample(vec3 in_dir, vec3 normal, float& pdf, curandState* rand_state);
	//__device__ vec3 specular_importance_sample(vec3 in_dir, vec3 normal, float& pdf);

	//__device__ vec3 specular_brdf(vec3 normal, vec3 in, vec3 out);
	__device__ vec3 lambertian_brdf(vec3 normal, vec3 out) const ;

	//__device__ float specular_pdf(vec3 sampled, vec3 reflected);
	__device__ float cosine_weighted_pdf(vec3 sampled, vec3 normal);
	//__device__ float schlick_fresnel(vec3 in, vec3 normal, float n1, float n2);

	//__device__ bool refract(vec3 in, vec3 normal, float n1, float n2, vec3& refract_dir);

private:
	__device__ vec3 to_world(const vec3& local, const vec3& normal);
};

__device__ inline vec3 material::lambertian_brdf(vec3 normal, vec3 out) const {
	if (dot(out, normal) > 0)
	{
		return kd / pi * dot(out, normal);
	}
	return { 0., 0.,0. };
}

__device__ vec3 material::cosine_weighted_sample(vec3 in_dir, vec3 normal, float& pdf, curandState* rand_state)  {
	float x1 = get_rand_float(rand_state);
	float x2 = get_rand_float(rand_state);
	float phi = 2 * pi * x2;
	float z = sqrt(x1);
	float sin_theta = sqrt(1.0 - z * z);
	vec3 local_ray_dir{ sin_theta * std::cos(phi), sin_theta * std::sin(phi), z };
	pdf = z / pi;
	return to_world(local_ray_dir, normal);
}

__device__ float material::cosine_weighted_pdf(vec3 sampled, vec3 normal) {
	return dot(sampled, normal) / pi;
}

__device__ vec3 material::to_world(const vec3& local, const vec3& normal) {
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
