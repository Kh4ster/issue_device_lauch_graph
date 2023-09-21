#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <chrono>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(-1);                                                              \
    }                                                                          \
}

__global__
void kernel_hat(int row)
{
    printf("Hello from %d\n", row);
}

__global__
void kernel(cudaGraphExec_t* graphExecs)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    cudaError_t status = cudaGraphLaunch(graphExecs[row], cudaStreamGraphFireAndForget);
    if (status != cudaSuccess) {
        printf("CUDA API failed for row %d with error: %s (%d)\n", row, cudaGetErrorString(status), status);
    }
    else
    {
        printf("Parent call to row %d worked\n", row);
    } 
}

int main()
{
    int blockSize = 128;

    std::vector<cudaGraphExec_t> graphExecs(blockSize);
    std::vector<cudaGraph_t> graphs(blockSize);
    cudaGraphExec_t* d_graphExecs;
    CHECK_CUDA(cudaMalloc(&d_graphExecs, blockSize * sizeof(cudaGraphExec_t)));
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Init child nodes
    for (int row = 0; row < blockSize; ++row)
    {
        dim3 grid(1);
        dim3 block(1);

        void* void_kernel_args[] = {
            (void*)&row
        };

        CHECK_CUDA(cudaGraphCreate(&graphs[row], 0));

        cudaKernelNodeParams kernelNodeParams;
        kernelNodeParams.func = (void*)kernel_hat;
        kernelNodeParams.gridDim = grid;
        kernelNodeParams.blockDim = block;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = void_kernel_args;
        kernelNodeParams.extra = nullptr;

        cudaGraphNode_t kernelNode;
        CHECK_CUDA(cudaGraphAddKernelNode(&kernelNode, graphs[row], nullptr, 0, &kernelNodeParams));
    
        CHECK_CUDA(cudaGraphInstantiate(&graphExecs[row], graphs[row], cudaGraphInstantiateFlagDeviceLaunch));
        CHECK_CUDA(cudaGraphUpload(graphExecs[row], stream));
    }
    cudaMemcpy(d_graphExecs, graphExecs.data(), blockSize * sizeof(cudaGraphExec_t), cudaMemcpyHostToDevice);
    // --

    // Init parent graph
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernel<<<1, blockSize, 0, stream>>>(d_graphExecs);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    cudaGraphUpload(instance, stream);
    // --

    // Launch
    CHECK_CUDA(cudaGraphLaunch(instance, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}