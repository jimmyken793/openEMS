#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "tools/constants.h"

typedef float4 CUDA_VECTOR;

inline void CheckCUDA(cudaError_t status, const char* operation)
{
	if (status != cudaSuccess)
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

inline size_t CUDAFieldIndex(unsigned int x, unsigned int y, unsigned int z,
		unsigned int numLinesY, unsigned int numLinesZ)
{
	return (static_cast<size_t>(x) * numLinesY + y) * numLinesZ + z;
}

inline size_t CUDAFieldCellCount(const unsigned int* numLines)
{
	return static_cast<size_t>(numLines[0]) * numLines[1] * numLines[2];
}

inline FDTD_FLOAT& CUDAComponent(CUDA_VECTOR& value, unsigned int component)
{
	return reinterpret_cast<FDTD_FLOAT*>(&value)[component];
}

inline const FDTD_FLOAT& CUDAComponent(const CUDA_VECTOR& value, unsigned int component)
{
	return reinterpret_cast<const FDTD_FLOAT*>(&value)[component];
}

#endif // CUDA_COMMON_H
