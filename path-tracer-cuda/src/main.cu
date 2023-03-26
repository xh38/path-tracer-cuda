#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "thrust/device_vector.h"
#include "thrust/device_malloc.h"
#include "thrust/host_vector.h"
#include <thrust/count.h>
#include <chrono>

#include "../include/scene.h"
#include "../include/renderer.h"
#include "../include/path_segment.h"
#include "../include/bvh.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

struct not_end {
	__host__ __device__	bool operator()(path_segment seg) {
        return seg.end == false;
	}
};

__host__ void write_result(std::string name, thrust::host_vector<vec3> frame_buffer, int width, int height) {
    float gamma = 2.2f;
    int len = width * height * 3;
    uint8_t* data = static_cast<uint8_t*>(std::malloc(len * sizeof(uint8_t)));
    uint8_t* raw = static_cast<uint8_t*>(std::malloc(len * sizeof(uint8_t)));

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                frame_buffer[i * width + j][k] = frame_buffer[i * width + j][k] > 1 ? 1.0f : frame_buffer[i * width + j][k];
                frame_buffer[i * width + j][k] = frame_buffer[i * width + j][k] < 0 ? 0.0f : frame_buffer[i * width + j][k];
                float correction = std::pow(frame_buffer[i * width + j][k], 1.0 / gamma);
                data[(i * width + j) * 3 + k] = static_cast<uint8_t>(correction * 255);
                raw[(i * width + j) * 3 + k] = static_cast<uint8_t>(frame_buffer[i * width + j][k] * 255);
                //std::cout << "x: " << j << " y: " << i << " c: " << k << "\nvalue: " << data[(i * width + j) * 3 + k] << "\n";
                //data[(i * width + j) * 3 + k] = static_cast<uint8_t>(frame_buffer[i * width + j][k] * 255);
            }
        }
    }
    std::string output_name = name + "_result.png";
    std::string raw_name = name + "_result_raw.png";
    stbi_write_png(output_name.c_str(), width, height, 3, data, width * 3);
    stbi_write_png(raw_name.c_str(), width, height, 3, raw, width * 3);
}

__global__ void rand_init(int width, int height, curandState* rand_state) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    //printf("pixel %d: rand init\n", x + y * width);
    const int pixel_index = y * width + x;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}


int main() {
    std::string name = R"(cornell-box)";
    const int spp = 512;
    int tx = 16;
    int ty = 16;
    const int max_depth = 5;
    //load scene host
    thrust::host_vector<material> host_material;
    thrust::host_vector<triangle> host_triangle;
    

    std::unique_ptr<camera_param> host_cam_param = std::make_unique<camera_param>();
    int num_lights;
    load_scene(name, host_triangle, host_material, host_cam_param, num_lights);
    //device_cam_param = thrust::device_ptr<camera_param>(host_cam_param.get());
    camera_param* device_cam_param;
    checkCudaErrors(cudaMallocManaged((void**)&device_cam_param, sizeof(camera_param)));
    device_cam_param->width = host_cam_param->width;
    device_cam_param->height = host_cam_param->height;
    device_cam_param->eye = host_cam_param->eye;
    device_cam_param->lookat = host_cam_param->lookat;
    device_cam_param->up = host_cam_param->up;
    device_cam_param->fovy = host_cam_param->fovy;
    printf("scene loaded\n");

    // set camera device
    thrust::device_ptr<camera> p_camera = thrust::device_malloc<camera>(1);
    set_camera_cuda<<<1,1>>>(p_camera.get(), device_cam_param);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "camera set\n";
    //TODO: bvh ACC
	//thrust::host_vector<lbvh_node> host_bvh;
    //build_bvh(host_bvh,  host_triangle);
    //std::cout << "bvh built\n" << std::endl;
	//copy triangle to device
    thrust::device_vector<triangle> dev_triangles(host_triangle);
    //thrust::device_vector<lbvh_node> dev_bvh(host_bvh);

    // get light indices
    thrust::device_vector<int> light_indices;
    light_indices.resize(num_lights);
    create_light_indices_cuda<<<1,1>>>(dev_triangles.data().get(), light_indices.data().get(), dev_triangles.size(), num_lights);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    std::cout << num_lights << " light indices created\n";

    thrust::device_ptr<mesh> p_mesh = thrust::device_malloc<mesh>(1);
	set_mesh_cuda<<<1,1>>>(p_mesh.get(), dev_triangles.data().get(), dev_triangles.size(), light_indices.data().get(), num_lights);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "mesh set\n";
    std::cout << "scene built on device!\n";

    int image_width = host_cam_param->width;
    int image_height = host_cam_param->height;
    std::cout << "rendering at res: " << image_width << "*" << image_height << std::endl;
    int pixel_num = image_width * image_height;

    thrust::device_vector<curandState> rand_state;
    thrust::device_vector<path_segment> path_segments;
    rand_state.resize(pixel_num);
    path_segments.resize(pixel_num);

    thrust::device_vector<vec3> device_frame_buffer;
    device_frame_buffer.resize(pixel_num);
    auto start = std::chrono::high_resolution_clock::now();
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    rand_init <<<blocks, threads>>>(image_width, image_height, rand_state.data().get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < spp; i++)
    {
        std::cout << "spp: " << i << "\n";
        thrust::device_vector<path_segment> dev_paths(pixel_num);
        generate_ray<<<blocks, threads>>>(dev_paths.data().get(), p_camera.get(), rand_state.data().get());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        int depth = 0;
        int path_remaining = pixel_num;
        while(path_remaining > 0 && depth < max_depth)
        {
            std::cout << "tracing depth: " << depth << "\n";
            std::cout << "path ramaining: " << path_remaining << "\n";
            compute_intersection<<<blocks, threads>>>(dev_paths.data().get(), p_mesh.get(), image_width, image_height);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
           // std::cout << "intersection get\n";

            shade_path_segment<<<blocks, threads>>>(dev_paths.data().get(), p_mesh.get(), image_width, image_height, rand_state.data().get(), depth);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            path_remaining = thrust::count_if(dev_paths.begin(), dev_paths.end(), not_end());
            depth++;
        }
        fill_frame_buffer<<<blocks, threads>>>(device_frame_buffer.data().get(), dev_paths.data().get(), image_width, image_height, spp);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "rendering spent: " << time.count() / 1000 << "ms." << std::endl;

    thrust::host_vector<vec3> host_frame_buffer = device_frame_buffer;
    write_result(name, host_frame_buffer, image_width, image_height);
    //write_result(name, host_frame_buffer, image_width, image_height);
}



