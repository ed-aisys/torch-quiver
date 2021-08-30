#pragma once
#include <torch/extension.h>
#include <stdio.h>

__device__ int find(const int64_t* offsets, const int device_count, const int64_t index){
    int i = 1;
    for(i = 1; i < device_count; i++){
        if(index < offsets[i]){
            return i - 1;
        }
    }
    return device_count - 1;
}
__global__ void quiver_tensor_gather(float** dev_ptrs, const int64_t* offsets, const int device_count,
                                     const int64_t* indices, int indice_length, 
                                     float* res,
                                     const int stride){

    // 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    unsigned int start = tid;
    int64_t dev_index = 0;
    int64_t dev_offset = 0; 
    float* dev_ptr = nullptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    unsigned int copy_count = 0;
    if(tid == 0){
    	printf("check tid = %d, start = %d, indices_length = %d \n", tid, start, indice_length);
	    printf("check offset[1] = %d, indices[10] = %d, dev_ptrs[0][1] = %d \n", offsets[1], indices[10], dev_ptrs[0][1]);

        dev_index = find(offsets, device_count, 49000);

        dev_offset = 90000 - offsets[dev_index];
        printf("index = %lld, dev_index = %lld, dev_offset = %lld \n", indices[start], dev_index, dev_offset);
    }
    __syncthreads();

    while(start < indice_length){
        dev_index = find(offsets, device_count, indices[start]);
        dev_ptr = dev_ptrs[dev_index];
        dev_offset = indices[start] - offsets[dev_index];
        src_copy_start = dev_offset * stride;
        dst_copy_start = start * stride;
        if(start == 0){
            printf("index = %lld, dev_index = %lld, dev_offset = %lld", indices[start], dev_index, dev_offset);

        }
        for(copy_count = 0; copy_count < stride; copy_count ++){
            res[dst_copy_start + copy_count] = dev_ptr[src_copy_start + copy_count];
        }
        start += step;
    }
}