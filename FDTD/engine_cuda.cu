#include "engine_cuda.h"
#include "operator_cuda.h"

#include "excitation.h"
#include "extensions/engine_ext_excitation.h"
#include "extensions/engine_ext_upml.h"
#include "extensions/engine_extension.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_upml.h"

#include <map>
#include <memory>
#include <sstream>
#include <vector>

struct CUDA_UPML_Data
{
	unsigned int start[3];
	unsigned int size[3];
	size_t cellCount;
	CUDA_VECTOR* vv;
	CUDA_VECTOR* vvfo;
	CUDA_VECTOR* vvfn;
	CUDA_VECTOR* ii;
	CUDA_VECTOR* iifo;
	CUDA_VECTOR* iifn;
	CUDA_VECTOR* voltFlux;
	CUDA_VECTOR* currFlux;
};

struct CUDA_Excitation_Entry
{
	unsigned short direction;
	FDTD_FLOAT amplitude;
	unsigned int delay;
};

struct CUDA_Excitation_Group
{
	size_t fieldIndex;
	unsigned int begin;
	unsigned int count;
};

struct CUDA_Excitation_Set
{
	CUDA_Excitation_Group* groups;
	CUDA_Excitation_Entry* entries;
	unsigned int groupCount;
	unsigned int entryCount;
};

struct CUDA_Excitation_Data
{
	FDTD_FLOAT* voltageSignal;
	FDTD_FLOAT* currentSignal;
	unsigned int signalLength;
	int fixedPeriod;
	CUDA_Excitation_Set voltage;
	CUDA_Excitation_Set current;
};

namespace
{
const unsigned int THREADS_PER_BLOCK = 256;

__host__ __device__ size_t FieldIndex(unsigned int x, unsigned int y, unsigned int z,
		unsigned int numLinesY, unsigned int numLinesZ)
{
	return (static_cast<size_t>(x) * numLinesY + y) * numLinesZ + z;
}

__device__ FDTD_FLOAT& VectorComponent(CUDA_VECTOR& value, unsigned int component)
{
	if (component == 0)
		return value.x;
	if (component == 1)
		return value.y;
	return value.z;
}

__global__ void VoltageKernel(CUDA_VECTOR* volt, const CUDA_VECTOR* curr,
		const CUDA_VECTOR* opvi, const CUDA_VECTOR* opvv,
		unsigned int numLinesX, unsigned int numLinesY, unsigned int numLinesZ,
		size_t cellCount)
{
	const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
	if (index >= cellCount)
		return;

	const size_t yz = static_cast<size_t>(numLinesY) * numLinesZ;
	const unsigned int x = static_cast<unsigned int>(index / yz);
	const size_t remainder = index - static_cast<size_t>(x) * yz;
	const unsigned int y = static_cast<unsigned int>(remainder / numLinesZ);
	const unsigned int z = static_cast<unsigned int>(remainder - static_cast<size_t>(y) * numLinesZ);
	if (x >= numLinesX)
		return;

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
		unsigned int numLinesX, unsigned int numLinesY, unsigned int numLinesZ,
		size_t cellCount)
{
	const size_t index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
	if (index >= cellCount)
		return;

	const size_t yz = static_cast<size_t>(numLinesY) * numLinesZ;
	const unsigned int x = static_cast<unsigned int>(index / yz);
	const size_t remainder = index - static_cast<size_t>(x) * yz;
	const unsigned int y = static_cast<unsigned int>(remainder / numLinesZ);
	const unsigned int z = static_cast<unsigned int>(remainder - static_cast<size_t>(y) * numLinesZ);
	if (x >= numLinesX - 1 || y >= numLinesY - 1 || z >= numLinesZ - 1)
		return;

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

__global__ void UPMLPreKernel(CUDA_VECTOR* field, CUDA_VECTOR* flux,
		const CUDA_VECTOR* sameField, const CUDA_VECTOR* oldFlux,
		unsigned int startX, unsigned int startY, unsigned int startZ,
		unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ,
		unsigned int globalY, unsigned int globalZ, size_t cellCount)
{
	const size_t localIndex = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
	if (localIndex >= cellCount)
		return;

	const size_t yz = static_cast<size_t>(sizeY) * sizeZ;
	const unsigned int x = static_cast<unsigned int>(localIndex / yz);
	const size_t remainder = localIndex - static_cast<size_t>(x) * yz;
	const unsigned int y = static_cast<unsigned int>(remainder / sizeZ);
	const unsigned int z = static_cast<unsigned int>(remainder - static_cast<size_t>(y) * sizeZ);
	const size_t globalIndex = FieldIndex(x + startX, y + startY, z + startZ, globalY, globalZ);

	const CUDA_VECTOR value = field[globalIndex];
	const CUDA_VECTOR previousFlux = flux[localIndex];
	const CUDA_VECTOR same = sameField[localIndex];
	const CUDA_VECTOR old = oldFlux[localIndex];
	CUDA_VECTOR nextFlux;
	nextFlux.x = same.x * value.x - old.x * previousFlux.x;
	nextFlux.y = same.y * value.y - old.y * previousFlux.y;
	nextFlux.z = same.z * value.z - old.z * previousFlux.z;
	nextFlux.w = 0;
	field[globalIndex] = previousFlux;
	flux[localIndex] = nextFlux;
}

__global__ void UPMLPostKernel(CUDA_VECTOR* field, CUDA_VECTOR* flux,
		const CUDA_VECTOR* newFlux,
		unsigned int startX, unsigned int startY, unsigned int startZ,
		unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ,
		unsigned int globalY, unsigned int globalZ, size_t cellCount)
{
	const size_t localIndex = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
	if (localIndex >= cellCount)
		return;

	const size_t yz = static_cast<size_t>(sizeY) * sizeZ;
	const unsigned int x = static_cast<unsigned int>(localIndex / yz);
	const size_t remainder = localIndex - static_cast<size_t>(x) * yz;
	const unsigned int y = static_cast<unsigned int>(remainder / sizeZ);
	const unsigned int z = static_cast<unsigned int>(remainder - static_cast<size_t>(y) * sizeZ);
	const size_t globalIndex = FieldIndex(x + startX, y + startY, z + startZ, globalY, globalZ);

	const CUDA_VECTOR previousFlux = flux[localIndex];
	const CUDA_VECTOR value = field[globalIndex];
	const CUDA_VECTOR coefficient = newFlux[localIndex];
	CUDA_VECTOR nextValue;
	nextValue.x = previousFlux.x + coefficient.x * value.x;
	nextValue.y = previousFlux.y + coefficient.y * value.y;
	nextValue.z = previousFlux.z + coefficient.z * value.z;
	nextValue.w = 0;
	flux[localIndex] = value;
	field[globalIndex] = nextValue;
}

__global__ void ExcitationKernel(CUDA_VECTOR* field,
		const CUDA_Excitation_Group* groups,
		const CUDA_Excitation_Entry* entries,
		unsigned int groupCount, const FDTD_FLOAT* signal,
		unsigned int signalLength, int fixedPeriod, int numTS)
{
	const unsigned int groupIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (groupIndex >= groupCount)
		return;

	const CUDA_Excitation_Group group = groups[groupIndex];
	CUDA_VECTOR value = field[group.fieldIndex];
	const int period = fixedPeriod > 0 ? fixedPeriod : numTS + 1;
	for (unsigned int offset = 0; offset < group.count; ++offset)
	{
		const CUDA_Excitation_Entry entry = entries[group.begin + offset];
		int excitationPosition = numTS - static_cast<int>(entry.delay);
		excitationPosition *= (excitationPosition > 0);
		excitationPosition %= period;
		excitationPosition *= (excitationPosition < static_cast<int>(signalLength));
		VectorComponent(value, entry.direction) += entry.amplitude * signal[excitationPosition];
	}
	field[group.fieldIndex] = value;
}

template <typename T>
T* AllocateAndCopy(const std::vector<T>& values, const char* operation)
{
	if (values.empty())
		return NULL;
	T* device = NULL;
	CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&device), values.size() * sizeof(T)), operation);
	try
	{
		CheckCUDA(cudaMemcpy(device, &values[0], values.size() * sizeof(T),
				cudaMemcpyHostToDevice), operation);
	}
	catch (...)
	{
		cudaFree(device);
		throw;
	}
	return device;
}

void FreeExcitationSet(CUDA_Excitation_Set& set)
{
	if (set.groups) cudaFree(set.groups);
	if (set.entries) cudaFree(set.entries);
	set.groups = NULL;
	set.entries = NULL;
	set.groupCount = 0;
	set.entryCount = 0;
}

void FreeUPML(CUDA_UPML_Data* data)
{
	if (!data)
		return;
	if (data->vv) cudaFree(data->vv);
	if (data->vvfo) cudaFree(data->vvfo);
	if (data->vvfn) cudaFree(data->vvfn);
	if (data->ii) cudaFree(data->ii);
	if (data->iifo) cudaFree(data->iifo);
	if (data->iifn) cudaFree(data->iifn);
	if (data->voltFlux) cudaFree(data->voltFlux);
	if (data->currFlux) cudaFree(data->currFlux);
	delete data;
}

void FreeExcitation(CUDA_Excitation_Data* data)
{
	if (!data)
		return;
	if (data->voltageSignal) cudaFree(data->voltageSignal);
	if (data->currentSignal) cudaFree(data->currentSignal);
	FreeExcitationSet(data->voltage);
	FreeExcitationSet(data->current);
	delete data;
}

void LaunchUPMLPre(CUDA_UPML_Data* data, CUDA_VECTOR* field, bool voltage,
		unsigned int globalY, unsigned int globalZ)
{
	const unsigned int blocks = static_cast<unsigned int>(
			(data->cellCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	UPMLPreKernel<<<blocks, THREADS_PER_BLOCK>>>(field,
			voltage ? data->voltFlux : data->currFlux,
			voltage ? data->vv : data->ii,
			voltage ? data->vvfo : data->iifo,
			data->start[0], data->start[1], data->start[2],
			data->size[0], data->size[1], data->size[2],
			globalY, globalZ, data->cellCount);
	CheckCUDA(cudaGetLastError(), voltage ?
			"CUDA UPML pre-voltage kernel launch failed" :
			"CUDA UPML pre-current kernel launch failed");
}

void LaunchUPMLPost(CUDA_UPML_Data* data, CUDA_VECTOR* field, bool voltage,
		unsigned int globalY, unsigned int globalZ)
{
	const unsigned int blocks = static_cast<unsigned int>(
			(data->cellCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	UPMLPostKernel<<<blocks, THREADS_PER_BLOCK>>>(field,
			voltage ? data->voltFlux : data->currFlux,
			voltage ? data->vvfn : data->iifn,
			data->start[0], data->start[1], data->start[2],
			data->size[0], data->size[1], data->size[2],
			globalY, globalZ, data->cellCount);
	CheckCUDA(cudaGetLastError(), voltage ?
			"CUDA UPML post-voltage kernel launch failed" :
			"CUDA UPML post-current kernel launch failed");
}

void LaunchExcitation(CUDA_Excitation_Data* data, CUDA_VECTOR* field,
		bool voltage, int numTS)
{
	const CUDA_Excitation_Set& set = voltage ? data->voltage : data->current;
	if (set.groupCount == 0)
		return;
	const unsigned int blocks =
			(set.groupCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	ExcitationKernel<<<blocks, THREADS_PER_BLOCK>>>(field, set.groups, set.entries,
			set.groupCount, voltage ? data->voltageSignal : data->currentSignal,
			data->signalLength, data->fixedPeriod, numTS);
	CheckCUDA(cudaGetLastError(), voltage ?
			"CUDA voltage excitation kernel launch failed" :
			"CUDA current excitation kernel launch failed");
}
}

Engine_CUDA* Engine_CUDA::New(const Operator_CUDA* op, unsigned int cudaDeviceNumber,
		bool deviceExtensions)
{
	cout << (deviceExtensions ?
			"Create FDTD engine (CUDA device extensions)" :
			"Create FDTD engine (CUDA lifecycle reference)") << endl;
	std::unique_ptr<Engine_CUDA> engine(new Engine_CUDA(op, deviceExtensions));
	engine->m_cudaDeviceNumber = cudaDeviceNumber;
	engine->Init();
	return engine.release();
}

Engine_CUDA::Engine_CUDA(const Operator_CUDA* op, bool deviceExtensions) : Engine(op),
	m_cudaOperator(op), m_cudaDeviceNumber(0), m_deviceExtensions(deviceExtensions),
	m_gridDim(0), m_blockDim(0), m_volt(NULL), m_curr(NULL)
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

void Engine_CUDA::FreeDeviceExtensions()
{
	for (size_t n = 0; n < m_cudaUPML.size(); ++n)
		FreeUPML(m_cudaUPML[n]);
	m_cudaUPML.clear();
	for (size_t n = 0; n < m_cudaExcitations.size(); ++n)
		FreeExcitation(m_cudaExcitations[n]);
	m_cudaExcitations.clear();
}

void Engine_CUDA::BuildDeviceExtensions()
{
	FreeDeviceExtensions();

	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		if (dynamic_cast<Engine_Ext_UPML*>(m_Eng_exts[n]) ||
				dynamic_cast<Engine_Ext_Excitation*>(m_Eng_exts[n]))
			continue;
		throw std::runtime_error("CUDA device extensions do not support active extension: " +
				m_Eng_exts[n]->GetExtensionName() + "; use --engine=cuda-reference");
	}

	try
	{
		for (size_t n = 0; n < m_Eng_exts.size(); ++n)
		{
			if (Engine_Ext_UPML* engineExtension =
					dynamic_cast<Engine_Ext_UPML*>(m_Eng_exts[n]))
			{
				Operator_Ext_UPML* op = engineExtension->m_Op_UPML;
				std::unique_ptr<CUDA_UPML_Data, void (*)(CUDA_UPML_Data*)> data(
					new CUDA_UPML_Data(), FreeUPML);
				data->start[0] = op->m_StartPos[0];
				data->start[1] = op->m_StartPos[1];
				data->start[2] = op->m_StartPos[2];
				data->size[0] = op->m_numLines[0];
				data->size[1] = op->m_numLines[1];
				data->size[2] = op->m_numLines[2];
				data->cellCount = static_cast<size_t>(data->size[0]) *
						data->size[1] * data->size[2];

				std::vector<CUDA_VECTOR> vv(data->cellCount);
				std::vector<CUDA_VECTOR> vvfo(data->cellCount);
				std::vector<CUDA_VECTOR> vvfn(data->cellCount);
				std::vector<CUDA_VECTOR> ii(data->cellCount);
				std::vector<CUDA_VECTOR> iifo(data->cellCount);
				std::vector<CUDA_VECTOR> iifn(data->cellCount);
				for (unsigned int x = 0; x < data->size[0]; ++x)
					for (unsigned int y = 0; y < data->size[1]; ++y)
						for (unsigned int z = 0; z < data->size[2]; ++z)
						{
							const size_t index = FieldIndex(x, y, z,
									data->size[1], data->size[2]);
							vv[index] = make_float4(op->vv[0][x][y][z],
									op->vv[1][x][y][z], op->vv[2][x][y][z], 0);
							vvfo[index] = make_float4(op->vvfo[0][x][y][z],
									op->vvfo[1][x][y][z], op->vvfo[2][x][y][z], 0);
							vvfn[index] = make_float4(op->vvfn[0][x][y][z],
									op->vvfn[1][x][y][z], op->vvfn[2][x][y][z], 0);
							ii[index] = make_float4(op->ii[0][x][y][z],
									op->ii[1][x][y][z], op->ii[2][x][y][z], 0);
							iifo[index] = make_float4(op->iifo[0][x][y][z],
									op->iifo[1][x][y][z], op->iifo[2][x][y][z], 0);
							iifn[index] = make_float4(op->iifn[0][x][y][z],
									op->iifn[1][x][y][z], op->iifn[2][x][y][z], 0);
						}

				data->vv = AllocateAndCopy(vv, "CUDA UPML vv copy failed");
				data->vvfo = AllocateAndCopy(vvfo, "CUDA UPML vvfo copy failed");
				data->vvfn = AllocateAndCopy(vvfn, "CUDA UPML vvfn copy failed");
				data->ii = AllocateAndCopy(ii, "CUDA UPML ii copy failed");
				data->iifo = AllocateAndCopy(iifo, "CUDA UPML iifo copy failed");
				data->iifn = AllocateAndCopy(iifn, "CUDA UPML iifn copy failed");
				const size_t bytes = data->cellCount * sizeof(CUDA_VECTOR);
				CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&data->voltFlux), bytes),
						"CUDA UPML voltage flux allocation failed");
				CheckCUDA(cudaMemset(data->voltFlux, 0, bytes),
						"CUDA UPML voltage flux initialization failed");
				CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&data->currFlux), bytes),
						"CUDA UPML current flux allocation failed");
				CheckCUDA(cudaMemset(data->currFlux, 0, bytes),
						"CUDA UPML current flux initialization failed");
				m_cudaUPML.push_back(data.get());
				data.release();
			}
			else if (Engine_Ext_Excitation* engineExtension =
					dynamic_cast<Engine_Ext_Excitation*>(m_Eng_exts[n]))
			{
				Operator_Ext_Excitation* op = engineExtension->m_Op_Exc;
				Excitation* excitation = op->m_Exc;
				std::unique_ptr<CUDA_Excitation_Data, void (*)(CUDA_Excitation_Data*)> data(
					new CUDA_Excitation_Data(), FreeExcitation);
				data->signalLength = excitation->GetLength();
				data->fixedPeriod = 0;
				if (excitation->GetSignalPeriod() > 0)
					data->fixedPeriod = static_cast<int>(
							excitation->GetSignalPeriod() / excitation->GetTimestep());
				if ((op->Volt_Count || op->Curr_Count) && data->signalLength == 0)
					throw std::runtime_error("CUDA excitation has no signal samples");

				std::vector<FDTD_FLOAT> voltageSignal(
						excitation->GetVoltageSignal(),
						excitation->GetVoltageSignal() + data->signalLength);
				std::vector<FDTD_FLOAT> currentSignal(
						excitation->GetCurrentSignal(),
						excitation->GetCurrentSignal() + data->signalLength);
				data->voltageSignal = AllocateAndCopy(voltageSignal,
						"CUDA voltage excitation signal copy failed");
				data->currentSignal = AllocateAndCopy(currentSignal,
						"CUDA current excitation signal copy failed");

				for (int kind = 0; kind < 2; ++kind)
				{
					const bool voltage = kind == 0;
					const unsigned int count = voltage ? op->Volt_Count : op->Curr_Count;
					std::map<size_t, std::vector<CUDA_Excitation_Entry> > grouped;
					for (unsigned int entry = 0; entry < count; ++entry)
					{
						const unsigned int x = voltage ? op->Volt_index[0][entry] : op->Curr_index[0][entry];
						const unsigned int y = voltage ? op->Volt_index[1][entry] : op->Curr_index[1][entry];
						const unsigned int z = voltage ? op->Volt_index[2][entry] : op->Curr_index[2][entry];
						CUDA_Excitation_Entry item;
						item.direction = voltage ? op->Volt_dir[entry] : op->Curr_dir[entry];
						item.amplitude = voltage ? op->Volt_amp[entry] : op->Curr_amp[entry];
						item.delay = voltage ? op->Volt_delay[entry] : op->Curr_delay[entry];
						grouped[FieldIndex(x, y, z, numLines[1], numLines[2])].push_back(item);
					}
					std::vector<CUDA_Excitation_Group> groups;
					std::vector<CUDA_Excitation_Entry> entries;
					for (std::map<size_t, std::vector<CUDA_Excitation_Entry> >::const_iterator
							it = grouped.begin(); it != grouped.end(); ++it)
					{
						CUDA_Excitation_Group group;
						group.fieldIndex = it->first;
						group.begin = static_cast<unsigned int>(entries.size());
						group.count = static_cast<unsigned int>(it->second.size());
						groups.push_back(group);
						entries.insert(entries.end(), it->second.begin(), it->second.end());
					}
					CUDA_Excitation_Set& set = voltage ? data->voltage : data->current;
					set.groupCount = static_cast<unsigned int>(groups.size());
					set.entryCount = static_cast<unsigned int>(entries.size());
					set.groups = AllocateAndCopy(groups, "CUDA excitation group copy failed");
					set.entries = AllocateAndCopy(entries, "CUDA excitation entry copy failed");
				}
				m_cudaExcitations.push_back(data.get());
				data.release();
			}
		}
		CheckCUDA(cudaDeviceSynchronize(), "CUDA device extension initialization failed");
	}
	catch (...)
	{
		FreeDeviceExtensions();
		throw;
	}
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

	const size_t cellCount = CUDAFieldCellCount(numLines);
	m_blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
	m_gridDim = dim3(static_cast<unsigned int>(
			(cellCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 1, 1);
	cout << "  CUDA linear block size: " << m_blockDim.x << endl;
	cout << "  CUDA linear grid size: " << m_gridDim.x << endl;

	FreeDeviceExtensions();
	FreeFields();
	ClearExtensions();
	try
	{
		m_volt = AllocateField("volt");
		m_curr = AllocateField("curr");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA field initialization failed");
		numTS = 0;
		InitExtensions();
		SortExtensionByPriority();
		if (m_deviceExtensions)
			BuildDeviceExtensions();
	}
	catch (...)
	{
		FreeDeviceExtensions();
		ClearExtensions();
		FreeFields();
		throw;
	}
}

void Engine_CUDA::Reset()
{
	FreeDeviceExtensions();
	FreeFields();
	ClearExtensions();
}

void Engine_CUDA::PrefetchDeviceData()
{
	const size_t bytes = CUDAFieldCellCount(numLines) * sizeof(CUDA_VECTOR);
	CheckCUDA(cudaMemPrefetchAsync(m_volt, bytes, m_cudaDeviceNumber),
			"CUDA voltage prefetch failed");
	CheckCUDA(cudaMemPrefetchAsync(m_curr, bytes, m_cudaDeviceNumber),
			"CUDA current prefetch failed");
	CheckCUDA(cudaMemPrefetchAsync(m_cudaOperator->m_vv, bytes, m_cudaDeviceNumber),
			"CUDA vv coefficient prefetch failed");
	CheckCUDA(cudaMemPrefetchAsync(m_cudaOperator->m_vi, bytes, m_cudaDeviceNumber),
			"CUDA vi coefficient prefetch failed");
	CheckCUDA(cudaMemPrefetchAsync(m_cudaOperator->m_ii, bytes, m_cudaDeviceNumber),
			"CUDA ii coefficient prefetch failed");
	CheckCUDA(cudaMemPrefetchAsync(m_cudaOperator->m_iv, bytes, m_cudaDeviceNumber),
			"CUDA iv coefficient prefetch failed");
}

bool Engine_CUDA::IterateReference(unsigned int iterTS)
{
	for (unsigned int iter = 0; iter < iterTS; ++iter)
	{
		DoPreVoltageUpdates();
		VoltageKernel<<<m_gridDim, m_blockDim>>>(m_volt, m_curr,
			m_cudaOperator->m_vi, m_cudaOperator->m_vv,
			numLines[0], numLines[1], numLines[2], CUDAFieldCellCount(numLines));
		CheckCUDA(cudaGetLastError(), "CUDA voltage kernel launch failed");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA voltage kernel execution failed");
		DoPostVoltageUpdates();
		Apply2Voltages();

		DoPreCurrentUpdates();
		CurrentKernel<<<m_gridDim, m_blockDim>>>(m_volt, m_curr,
			m_cudaOperator->m_iv, m_cudaOperator->m_ii,
			numLines[0], numLines[1], numLines[2], CUDAFieldCellCount(numLines));
		CheckCUDA(cudaGetLastError(), "CUDA current kernel launch failed");
		CheckCUDA(cudaDeviceSynchronize(), "CUDA current kernel execution failed");
		DoPostCurrentUpdates();
		Apply2Current();

		++numTS;
	}
	return true;
}

bool Engine_CUDA::IterateDevice(unsigned int iterTS)
{
	PrefetchDeviceData();
	const size_t cellCount = CUDAFieldCellCount(numLines);
	for (unsigned int iter = 0; iter < iterTS; ++iter)
	{
		for (std::vector<CUDA_UPML_Data*>::reverse_iterator it = m_cudaUPML.rbegin();
				it != m_cudaUPML.rend(); ++it)
			LaunchUPMLPre(*it, m_volt, true, numLines[1], numLines[2]);

		VoltageKernel<<<m_gridDim, m_blockDim>>>(m_volt, m_curr,
			m_cudaOperator->m_vi, m_cudaOperator->m_vv,
			numLines[0], numLines[1], numLines[2], cellCount);
		CheckCUDA(cudaGetLastError(), "CUDA voltage kernel launch failed");

		for (size_t n = 0; n < m_cudaUPML.size(); ++n)
			LaunchUPMLPost(m_cudaUPML[n], m_volt, true, numLines[1], numLines[2]);
		for (size_t n = 0; n < m_cudaExcitations.size(); ++n)
			LaunchExcitation(m_cudaExcitations[n], m_volt, true, static_cast<int>(numTS));

		for (std::vector<CUDA_UPML_Data*>::reverse_iterator it = m_cudaUPML.rbegin();
				it != m_cudaUPML.rend(); ++it)
			LaunchUPMLPre(*it, m_curr, false, numLines[1], numLines[2]);

		CurrentKernel<<<m_gridDim, m_blockDim>>>(m_volt, m_curr,
			m_cudaOperator->m_iv, m_cudaOperator->m_ii,
			numLines[0], numLines[1], numLines[2], cellCount);
		CheckCUDA(cudaGetLastError(), "CUDA current kernel launch failed");

		for (size_t n = 0; n < m_cudaUPML.size(); ++n)
			LaunchUPMLPost(m_cudaUPML[n], m_curr, false, numLines[1], numLines[2]);
		for (size_t n = 0; n < m_cudaExcitations.size(); ++n)
			LaunchExcitation(m_cudaExcitations[n], m_curr, false, static_cast<int>(numTS));

		++numTS;
	}
	CheckCUDA(cudaDeviceSynchronize(), "CUDA device iteration failed");
	return true;
}

bool Engine_CUDA::IterateTS(unsigned int iterTS)
{
	return m_deviceExtensions ? IterateDevice(iterTS) : IterateReference(iterTS);
}
