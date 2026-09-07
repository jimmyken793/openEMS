#include "operator_cuda.h"
#include "engine_cuda.h"

#include <sstream>
#include <memory>

Operator_CUDA* Operator_CUDA::New(unsigned int cudaDeviceNumber, bool deviceExtensions)
{
	cout << (deviceExtensions ? "Create FDTD operator (CUDA device extensions)" :
			"Create FDTD operator (CUDA reference)") << endl;
	std::unique_ptr<Operator_CUDA> op(new Operator_CUDA(deviceExtensions));
	op->m_cudaDeviceNumber = cudaDeviceNumber;
	op->Init();
	return op.release();
}

Operator_CUDA::Operator_CUDA(bool deviceExtensions) : Operator(),
	m_cudaDeviceNumber(0), m_deviceExtensions(deviceExtensions), m_vv(NULL), m_vi(NULL), m_ii(NULL), m_iv(NULL)
{
	static_assert(sizeof(CUDA_VECTOR) == sizeof(FDTD_FLOAT) * 4,
		"CUDA_VECTOR must contain four FDTD_FLOAT values");
}

Operator_CUDA::~Operator_CUDA()
{
	FreeCoefficients();
}

Engine* Operator_CUDA::CreateEngine()
{
	m_Engine = Engine_CUDA::New(this, m_cudaDeviceNumber, m_deviceExtensions);
	return m_Engine;
}

CUDA_VECTOR* Operator_CUDA::AllocateCoefficients(const char* name)
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
		CheckCUDA(status, "CUDA coefficient initialization failed");
	}
	return values;
}

void Operator_CUDA::FreeCoefficients()
{
	if (m_vv) cudaFree(m_vv);
	if (m_vi) cudaFree(m_vi);
	if (m_ii) cudaFree(m_ii);
	if (m_iv) cudaFree(m_iv);
	m_vv = m_vi = m_ii = m_iv = NULL;
}

void Operator_CUDA::InitOperator()
{
	FreeCoefficients();
	CheckCUDA(cudaSetDevice(m_cudaDeviceNumber), "Unable to select CUDA device");
	try
	{
		m_vv = AllocateCoefficients("vv");
		m_vi = AllocateCoefficients("vi");
		m_ii = AllocateCoefficients("ii");
		m_iv = AllocateCoefficients("iv");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA coefficient initialization failed");
	}
	catch (...)
	{
		FreeCoefficients();
		throw;
	}
}

void Operator_CUDA::Reset()
{
	FreeCoefficients();
	Operator::Reset();
}
