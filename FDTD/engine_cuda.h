#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include "cuda_common.h"
#include "engine.h"

class Operator_CUDA;

class Engine_CUDA : public Engine
{
public:
	static Engine_CUDA* New(const Operator_CUDA* op, unsigned int cudaDeviceNumber);
	virtual ~Engine_CUDA();

	virtual void Init();
	virtual void Reset();
	virtual bool IterateTS(unsigned int iterTS);

	virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const { return CUDAComponent(m_volt[Index(x,y,z)], n); }
	virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const { return GetVolt(n, pos[0], pos[1], pos[2]); }
	virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const { return CUDAComponent(m_curr[Index(x,y,z)], n); }
	virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const { return GetCurr(n, pos[0], pos[1], pos[2]); }

	virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) { CUDAComponent(m_volt[Index(x,y,z)], n) = value; }
	virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value) { SetVolt(n, pos[0], pos[1], pos[2], value); }
	virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) { CUDAComponent(m_curr[Index(x,y,z)], n) = value; }
	virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value) { SetCurr(n, pos[0], pos[1], pos[2], value); }

protected:
	Engine_CUDA(const Operator_CUDA* op);

private:
	size_t Index(unsigned int x, unsigned int y, unsigned int z) const { return CUDAFieldIndex(x, y, z, numLines[1], numLines[2]); }
	void FreeFields();
	CUDA_VECTOR* AllocateField(const char* name);

	const Operator_CUDA* m_cudaOperator;
	unsigned int m_cudaDeviceNumber;
	dim3 m_gridDim;
	dim3 m_blockDim;
	CUDA_VECTOR* m_volt;
	CUDA_VECTOR* m_curr;
};

#endif // ENGINE_CUDA_H
