#pragma once

#include <vector>
#include <array>

#include "camera.h"
#include "triangle.h"
#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/device_make_unique.h"
#include "mesh.h"
#include "bvh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../3rdparty/tiny_obj_loader.h"
#include "../3rdparty/tinyxml2.h"


__host__ void
load_materials(thrust::host_vector<material> &scene_m, const std::vector<tinyobj::material_t> &loaded_m,
               const std::map<std::string, vec3> &light_map, std::vector<int>& light_material_indices, std::string prefix) {
    for (int i = 0 ; i < loaded_m.size(); i++) {
        tinyobj::material_t m = loaded_m[i];
        material temp_m{};
        temp_m.kd = {m.diffuse[0], m.diffuse[1], m.diffuse[2]};
        temp_m.ks = {m.specular[0], m.specular[1], m.specular[2]};
        temp_m.tr = {m.transmittance[0], m.transmittance[1], m.transmittance[2]};
        temp_m.ni = m.ior;
        temp_m.ns = m.shininess;
        //if (m.diffuse_texname != "")
        //{
        //	std::string path = prefix + m.diffuse_texname;
        //	temp_m.texture = Texture(path);
        //	temp_m.hasTexture = true;
        //}
        if (light_map.count(m.name) != 0) {
            temp_m.hasEmit = true;
            temp_m.radiance = light_map.at(m.name);
            light_material_indices.push_back(i);
        }
        scene_m.push_back(temp_m);
    }
}

__host__ bool is_matelial_light(std::vector<int>& light_indeces, int material_index) {
    for(int i : light_indeces) {
        if (i == material_index) {
            return true;
        }
    }
    return false;
}

__host__ void load_scene(std::string name, thrust::host_vector<triangle> &host_triangles, thrust::host_vector<material> &host_materials, 
    thrust::host_vector<int>& host_light_indices, std::unique_ptr<camera_param>& cam_param)
{
    std::string prefix = R"(C:\Users\AAA\Desktop\PathTracer\PathTracer\PathTracer\example-scenes-cg22\)";
    const std::string &scene_name = name;
    std::string scene_filename = prefix + scene_name + "\\" + scene_name + ".xml";
    std::cout << "loading scene file " << scene_filename << std::endl;
    tinyxml2::XMLDocument doc;
    doc.LoadFile(scene_filename.c_str());

    //camera
    tinyxml2::XMLElement *camera_element = doc.FirstChildElement("camera");

    float fovy = camera_element->FloatAttribute("fovy");
    int width = camera_element->IntAttribute("width");
    int height = camera_element->IntAttribute("height");

    //eye
    tinyxml2::XMLElement *eye_element = camera_element->FirstChildElement("eye");
    float eye_x = eye_element->FloatAttribute("x");
    float eye_y = eye_element->FloatAttribute("y");
    float eye_z = eye_element->FloatAttribute("z");
    vec3 eye(eye_x, eye_y, eye_z);

    //lookat
    tinyxml2::XMLElement *lookat_element = camera_element->FirstChildElement("lookat");
    float lookat_x = lookat_element->FloatAttribute("x");
    float lookat_y = lookat_element->FloatAttribute("y");
    float lookat_z = lookat_element->FloatAttribute("z");
    vec3 lookat(lookat_x, lookat_y, lookat_z);

    //up
    tinyxml2::XMLElement *up_element = camera_element->FirstChildElement("up");
    float up_x = up_element->FloatAttribute("x");
    float up_y = up_element->FloatAttribute("y");
    float up_z = up_element->FloatAttribute("z");
    vec3 up(up_x, up_y, up_z);

    cam_param->width = width;
    cam_param->height = height;
    cam_param->eye = eye;
    cam_param->lookat = lookat;
    cam_param->up = up;
    cam_param->fovy = fovy;

    std::cout << "camera loaded" << std::endl;

    //lights
    std::map<std::string, vec3> light_map;
    tinyxml2::XMLElement *light_element = doc.FirstChildElement("light");

    while (light_element != nullptr) {
        std::string mtl_name(light_element->Attribute("mtlname"));

        std::istringstream radiance_string(light_element->Attribute("radiance"));
        vec3 radiance;
        for (float &i: radiance.e) {
            std::string temp;
            std::getline(radiance_string, temp, ',');
            i = atof(temp.c_str());
        }
        light_map.insert(std::make_pair(mtl_name, radiance));
        light_element = light_element->NextSiblingElement("light");
    }
    std::cout << light_map.size() << " lights radiance loaded" << std::endl;

    //materials
    std::string obj_filename = prefix + scene_name + "\\" + scene_name + ".obj";
    std::string mtl_folder = prefix + scene_name;

    std::cout << "loading obj file " << obj_filename << std::endl;
    std::cout << "loading mtl in " << mtl_folder << std::endl;

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReaderConfig config;
    reader_config.mtl_search_path = mtl_folder; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(obj_filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    std::vector<int> light_material_indices;
    std::string texture_prefix = prefix + scene_name + "\\";
    load_materials(host_materials, materials, light_map, light_material_indices, texture_prefix);
    std::cout << materials.size() << " materials loaded" << std::endl;

    size_t tri_id = 0;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            const int material_index = shapes[s].mesh.material_ids[f];
            
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            std::array<vec3, 3> face_vertices;
            std::array<float, 3> tex_coord_x;
            std::array<float, 3> tex_coord_y;
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                face_vertices[v] = vec3(vx, vy, vz);
                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                }

            }
            triangle temp_triangle{face_vertices[0], face_vertices[1], face_vertices[2], material_index, tri_id};
            host_triangles.push_back(temp_triangle);
            if (is_matelial_light(light_material_indices, material_index)) {
                host_light_indices.push_back(tri_id);
            }
            tri_id++;
            index_offset += fv;
        }
    }
    std::cout << host_triangles.size() << " triangles loaded" << std::endl;
}

__host__ void build_bvh(thrust::host_vector<lbvh_node>& host_bvh, thrust::host_vector<triangle>& host_triangle) {
    rbvh_node* root = recurisve_build(host_triangle);
    size_t num_nodes = root->count();
    std::cout << "rec build done, node: " << num_nodes << std::endl;
    // make rbvh to lbvh
    host_bvh.resize(2*num_nodes-1);
    size_t next_free_node = 0;
    size_t node_index = 0;
    std::queue<rbvh_node*> rnode_queue = std::queue<rbvh_node*>();
    rnode_queue.push(root);
    while(!rnode_queue.empty())
    {
        //std::cout << "setting node: " << node_index << std::endl;
        //std::cout << "queue size: " << rnode_queue.size() << std::endl;
        rbvh_node* rnode = rnode_queue.front();
        lbvh_node& cur_node = host_bvh[node_index];
        cur_node.tri_id = rnode->object.m_id;
        cur_node.box = rnode->box;
        cur_node.is_leaf = rnode->is_leaf;

        if(rnode->left != nullptr)
        {
	        cur_node.left = ++next_free_node;
			rnode_queue.push(rnode->left);
		}
		else
		{
			cur_node.left = -1;
        }
        if(rnode->right != nullptr)
        {
            cur_node.right = ++next_free_node;
            rnode_queue.push(rnode->right);
        }
        else
        {
	        cur_node.right = -1;
        }

        node_index++;
        rnode_queue.pop();
    }

}


__global__ void set_camera_cuda(camera* dev_cam, camera_param* cam_param) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x ==0 && y == 0)dev_cam->set(cam_param);
}

//__global__ void create_light_indices_cuda(triangle* triangles, int* lights, int num_triangle, int num_lights) {
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	if (x == 0&& y== 0)
//	{
//        for (int i = 0, light_count = 0; i < num_triangle && light_count < num_lights; i++) {
//            if (triangles[i].m_material.is_light()) {
//                lights[light_count] = i;
//                light_count++;
//            }
//        }
//	}
//}

__global__ void set_mesh_cuda(mesh* p_mesh, triangle* triangles, int num_t, int* light_indices, int num_lights, material* materials, int num_material) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 && y == 0)
	{
		p_mesh->set(num_t, num_lights, num_material, triangles, light_indices, materials);
	}
}

//__global__ void set_BVH_mesh_cuda(mesh* p_mesh, triangle* triangles, int num_t, int* light_indices, int num_lights) {
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    if (x == 0 && y == 0)
//    {
//        p_mesh->set(num_t, num_lights, triangles, light_indices);
//    }
//}
