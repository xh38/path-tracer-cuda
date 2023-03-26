#pragma once
#include <queue>

#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "AABB.h"
#include "triangle.h"

struct rbvh_node {
	AABB box;
	rbvh_node* left;
	rbvh_node* right;
	bool is_leaf;
	int tri_id;
	triangle object;

	int count() {
		if (is_leaf)
		{
			return 1;
		}
		return left->count() + right->count();
	}
};

struct lbvh_node {
	AABB box;
	int left;
	int right;
	bool is_leaf;
	int tri_id;
};

rbvh_node* recurisve_build(thrust::host_vector<triangle>& objects) {
	auto node = new rbvh_node{};

	AABB aabb;
	for (auto object : objects)
	{
		aabb = merge(aabb, object.get_aabb());
	}

	if (objects.size() == 1)
	{
		node->box = objects[0].get_aabb();
		node->object = objects[0];
		node->left = nullptr;
		node->right = nullptr;
		node->is_leaf = true;
		//node->num = 1;
	}
	else if (objects.size() == 2)
	{
		thrust::host_vector<triangle> left;
		left.push_back(objects[0]);

		thrust::host_vector<triangle> right;
		right.push_back(objects[0]);

		node->left = recurisve_build(left);
		node->right = recurisve_build(right);
		node->box = merge(node->left->box, node->right->box);
		node->is_leaf = false;
		//node->num = 2;
	}
	else
	{
		//node->num = objects.size();
		const int max_extent = aabb.get_max_extent();
		if (max_extent == 0)
		{
			thrust::sort(objects.begin(), objects.end(), [](triangle obj_1, triangle obj_2) {return obj_1.get_aabb().get_middle(0) < obj_2.get_aabb().get_middle(0); });
		}
		else if (max_extent == 1)
		{
			thrust::sort(objects.begin(), objects.end(), [](triangle obj_1, triangle obj_2) {return obj_1.get_aabb().get_middle(1) < obj_2.get_aabb().get_middle(1); });
		}
		else if (max_extent == 2)
		{
			thrust::sort(objects.begin(), objects.end(), [](triangle obj_1, triangle obj_2) {return obj_1.get_aabb().get_middle(2) < obj_2.get_aabb().get_middle(2); });
		}
		else
		{
			throw std::runtime_error("only three");
		}

		auto begin = objects.begin();
		auto end = objects.end();
		auto mid = begin + objects.size() / 2;

		auto left_objects = thrust::host_vector<triangle>(begin, mid);
		auto right_objects = thrust::host_vector<triangle>(mid, end);

		node->left = recurisve_build(left_objects);
		node->right = recurisve_build(right_objects);
		node->box = aabb;
		node->is_leaf = false;
	}
	return node;
}
