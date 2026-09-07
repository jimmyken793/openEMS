#include "engine_cuda.h"
#include "operator_cuda.h"

#include <sstream>
#include <memory>

namespace
{
__host__ __device__ size_t FieldIndex(unsigned int x, unsigned int y, unsigned int z,
		unsigned int numLinesY, unsigned int numLinesZ)
{
	return (static_cast<size_t>(x) * numLinesY + y) * numLinesZ + z;
}

__global__ void VoltageKernel(CUDA_VECTOR* volt, const CUDA_VECTOR* curr,
		const CUDA_VECTOR* opvi, const CUDA_VECTOR* opvv,
		unsigned int numLinesX, unsigned int numLinesY, unsigned int numLinesZ)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x >= numLinesX || y >= numLinesY || z >= numLinesZ)
		return;

	const size_t index = FieldIndex(x, y, z, numLinesY, numLinesZ);
	const size_t indexX = FieldIndex(x - (x != 0), y, z, numLinesY, numLinesZ);
	const size_t indexY = FieldIndex(x, y - (y != 0), z, numLinesY, numLinesZ);
	const size_t indexZ = FieldIndex(x, y, z - (z != 0), numLinesY, numLinesZ);

	CUDA_VECTOR value = volt[index];
	const CUDA_VECTOR current = curr[index];
	const CUDA_VECTOR currentX = curr[indexX];
	const CUDA_VECTOR currentY = curr[indexY];
	const CUDA_VECTOR currentZ = curr[indexZ];
	const CUDA_VECTOR vi = opvi[index];
	const CUDA_VECTOR vv = opvv[index];

	value.x = value.x * vv.x + vi.x * (current.z - currentY.z - current.y + currentZ.y);
	value.y = value.y * vv.y + vi.y * (current.x - currentZ.x - current.z + currentX.z);
	value.z = value.z * vv.z + vi.z * (current.y - currentX.y - current.x + currentY.x);
	volt[index] = value;
}

__global__ void CurrentKernel(const CUDA_VECTOR* volt, CUDA_VECTOR* curr,
		const CUDA_VECTOR* opiv, const CUDA_VECTOR* opii,
		unsigned int numLinesX, unsigned int numLinesY, unsigned int numLinesZ)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x >= numLinesX - 1 || y >= numLinesY - 1 || z >= numLinesZ - 1)
		return;

	const size_t index = FieldIndex(x, y, z, numLinesY, numLinesZ);
	const size_t indexX = FieldIndex(x + 1, y, z, numLinesY, numLinesZ);
	const size_t indexY = FieldIndex(x, y + 1, z, numLinesY, numLinesZ);
	const size_t indexZ = FieldIndex(x, y, z + 1, numLinesY, numLinesZ);

	CUDA_VECTOR value = curr[index];
	const CUDA_VECTOR voltage = volt[index];
	const CUDA_VECTOR voltageX = volt[indexX];
	const CUDA_VECTOR voltageY = volt[indexY];
	const CUDA_VECTOR voltageZ = volt[indexZ];
	const CUDA_VECTOR iv = opiv[index];
	const CUDA_VECTOR ii = opii[index];

	value.x = value.x * ii.x + iv.x * (voltage.z - voltageY.z - voltage.y + voltageZ.y);
	value.y = value.y * ii.y + iv.y * (voltage.x - voltageZ.x - voltage.z + voltageX.z);
	value.z = value.z * ii.z + iv.z * (voltage.y - voltageX.y - voltage.x + voltageY.x);
	curr[index] = value;
}
}

Engine_CUDA* Engine_CUDA::New(const Operator_CUDA* op, unsigned int cudaDeviceNumber)
{
	cout << "Create FDTD engine (CUDA lifecycle reference)" << endl;
	std::unique_ptr<Engine_CUDA> engine(new Engine_CUDA(op));
	engine->m_cudaDeviceNumber = cudaDeviceNumber;
	engine->Init();
	return engine.release();
}

Engine_CUDA::Engine_CUDA(const Operator_CUDA* op) : Engine(op),
	m_cudaOperator(op), m_cudaDeviceNumber(0), m_gridDim(0), m_blockDim(0),
	m_volt(NULL), m_curr(NULL)
{
	m_type = CUDA;
}

Engine_CUDA::~Engine_CUDA()
{
	Reset();
}

CUDA_VECTOR* Engine_CUDA::AllocateField(const char* name)
{
	CUDA_VECTOR* values = NULL;
	const size_t bytes = CUDAFieldCellCount(numLines) * sizeof(CUDA_VECTOR);
	cudaError_t status = cudaMallocManaged(reinterpret_cast<void**>(&values), bytes);
	if (status != cudaSuccess)
	{
		std::ostringstream message;
		message << "CUDA allocation failed for " << name << " (" << bytes << " bytes)";
		CheckCUDA(status, message.str().c_str());
	}
	status = cudaMemset(values, 0, bytes);
	if (status != cudaSuccess)
	{
		cudaFree(values);
		CheckCUDA(status, "CUDA field initialization failed");
	}
	return values;
}

void Engine_CUDA::FreeFields()
{
	if (m_volt) cudaFree(m_volt);
	if (m_curr) cudaFree(m_curr);
	m_volt = m_curr = NULL;
}

void Engine_CUDA::Init()
{
	int deviceCount = 0;
	CheckCUDA(cudaGetDeviceCount(&deviceCount), "Unable to query CUDA devices");
	if (deviceCount <= 0)
		throw std::runtime_error("No CUDA devices found");
	if (m_cudaDeviceNumber >= static_cast<unsigned int>(deviceCount))
		throw std::runtime_error("CUDA device number out of range");
	CheckCUDA(cudaSetDevice(m_cudaDeviceNumber), "Unable to select CUDA device");

	cudaDeviceProp properties;
	CheckCUDA(cudaGetDeviceProperties(&properties, m_cudaDeviceNumber),
		"Unable to query CUDA device properties");
	cout << "  Running on device " << m_cudaDeviceNumber << ": " << properties.name << endl;

	m_blockDim = dim3(8, 8, 4);
	m_gridDim = dim3(
		(numLines[0] + m_blockDim.x - 1) / m_blockDim.x,
		(numLines[1] + m_blockDim.y - 1) / m_blockDim.y,
		(numLines[2] + m_blockDim.z - 1) / m_blockDim.z);
	cout << "  CUDA block dimensions: " << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << endl;
	cout << "  CUDA grid dimensions: " << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << endl;

	FreeFields();
	ClearExtensions();
	try
	{
		m_volt = AllocateField("volt");
		m_curr = AllocateField("curr");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA field initialization failed");
	}
	catch (...)
	{
		FreeFields();
		throw;
	}

	numTS = 0;
	InitExtensions();
	SortExtensionByPriority();
}

void Engine_CUDA::Reset()
{
	FreeFields();
	ClearExtensions();
}

bool Engine_CUDA::IterateTS(unsigned int iterTS)
{
	for (unsigned int iter = 0; iter < iterTS; ++iter)
	{
		DoPreVoltageUpdates();
		VoltageKernel<<<m_gridDim, m_blockDim>>>(m_volt, m_curr,
			m_cudaOperator->m_vi, m_cudaOperator->m_vv,
			numLines[0], numLines[1], numLines[2]);
		CheckCUDA(cudaGetLastError(), "CUDA voltage kernel launch failed");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA voltage kernel execution failed");
		DoPostVoltageUpdates();
		Apply2Voltages();

		DoPreCurrentUpdates();
		CurrentKernel<<<m_gridDim, m_blockDim>>>(m_volt, m_curr,
			m_cudaOperator->m_iv, m_cudaOperator->m_ii,
			numLines[0], numLines[1], numLines[2]);
		CheckCUDA(cudaGetLastError(), "CUDA current kernel launch failed");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA current kernel execution failed");
		DoPostCurrentUpdates();
		Apply2Current();

		++numTS;
	}
	return true;
}
