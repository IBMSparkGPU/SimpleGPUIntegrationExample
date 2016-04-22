/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "utilities.h"

// kernel to increment values
__global__ void incrementKernel(int len, int *input, int *output)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    
    for (int pos = tid; pos < len; pos += threadN)
    	output[pos] = input[pos] + 1;
}

// returns 0 on success
extern "C" int gpu_increment(int partitionId, int len, int* input, int* output) {
	int deviceID = get_gpu();
	if (deviceID == -1)
		return 1; // no device present
	
	cudaDeviceProp deviceProp;
	if (cudaSetDevice(deviceID) || cudaGetDeviceProperties(&deviceProp, deviceID)) {
		fprintf(stderr, "Cuda error in SetDevice\n");
		return 1;
	}

	fprintf(stderr,
			"gpu_test(): Partition %d will be executed on GPU %d\n",
			partitionId, deviceID);
	
	// register host memory in the GPU space
	// note that we pinned these buffers in the JNI code already
	if (cudaHostRegister(input, len*sizeof(int), 0) || cudaHostRegister(output, len*sizeof(int), 0)) {
		fprintf(stderr, "Unable to register data buffer: %s: %s\n",  cudaGetErrorName(cudaPeekAtLastError()), cudaGetErrorString(cudaGetLastError()));
		return 1;
	}
	
	// get pointers valid from the device 
	int *d_input, *d_output;
	if (cudaHostGetDevicePointer((void **)&d_input, input, 0) || cudaHostGetDevicePointer((void **)&d_output, output, 0)) {
		fprintf(stderr, "Unable to get device pointer to host memory: %s: %s\n",  cudaGetErrorName(cudaPeekAtLastError()), cudaGetErrorString(cudaGetLastError()));
		return 1;
	}
	
	// invoke kernel
	int threadsPerBlock = 256; 
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(len, d_input, d_output);
	cudaDeviceSynchronize();
	if (cudaPeekAtLastError()) {
		fprintf(stderr, "Unable to invoke kernel: %s\n",  cudaGetErrorString(cudaGetLastError()));
		return 1;
	}
		
	// unregister host memory
	if (cudaHostUnregister(input) || cudaHostUnregister(output)) {
		fprintf(stderr, "Unable to unregister host memory: %s: %s\n",  cudaGetErrorName(cudaPeekAtLastError()),  cudaGetErrorString(cudaGetLastError()));
		return 1;
	}
	
	return 0;
}

