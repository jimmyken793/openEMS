#ifndef OPERATOR_CUDA_H
#define OPERATOR_CUDA_H

#include "cuda_common.h"
#include "operator.h"

class Operator_CUDA : public Operator
{
	friend class Engine_CUDA;

public:
	static Operator_CUDA* New(unsigned int cudaDeviceNumber = 0, bool deviceExtensions = true);
	virtual ~Operator_CUDA();

	virtual Engine* CreateEngine();
	virtual void Reset();

	virtual FDTD_FLOAT GetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const { return CUDAComponent(m_vv[Index(x,y,z)], n); }
	virtual FDTD_FLOAT GetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const { return CUDAComponent(m_vi[Index(x,y,z)], n); }
	virtual FDTD_FLOAT GetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const { return CUDAComponent(m_ii[Index(x,y,z)], n); }
	virtual FDTD_FLOAT GetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const { return CUDAComponent(m_iv[Index(x,y,z)], n); }
	virtual FDTD_FLOAT GetVV(unsigned int n, unsigned int pos[3]) const { return GetVV(n, pos[0], pos[1], pos[2]); }
	virtual FDTD_FLOAT GetVI(unsigned int n, unsigned int pos[3]) const { return GetVI(n, pos[0], pos[1], pos[2]); }
	virtual FDTD_FLOAT GetII(unsigned int n, unsigned int pos[3]) const { return GetII(n, pos[0], pos[1], pos[2]); }
	virtual FDTD_FLOAT GetIV(unsigned int n, unsigned int pos[3]) const { return GetIV(n, pos[0], pos[1], pos[2]); }

	virtual void SetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) { CUDAComponent(m_vv[Index(x,y,z)], n) = value; }
	virtual void SetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) { CUDAComponent(m_vi[Index(x,y,z)], n) = value; }
	virtual void SetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) { CUDAComponent(m_ii[Index(x,y,z)], n) = value; }
	virtual void SetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value) { CUDAComponent(m_iv[Index(x,y,z)], n) = value; }

protected:
	Operator_CUDA(bool deviceExtensions);
	virtual void InitOperator();

private:
	size_t Index(unsigned int x, unsigned int y, unsigned int z) const { return CUDAFieldIndex(x, y, z, numLines[1], numLines[2]); }
	void FreeCoefficients();
	CUDA_VECTOR* AllocateCoefficients(const char* name);

	unsigned int m_cudaDeviceNumber;
	bool m_deviceExtensions;
	CUDA_VECTOR* m_vv;
	CUDA_VECTOR* m_vi;
	CUDA_VECTOR* m_ii;
	CUDA_VECTOR* m_iv;
};

#endif // OPERATOR_CUDA_H
