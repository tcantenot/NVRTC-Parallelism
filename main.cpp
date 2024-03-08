#include <nvrtc.h>
#include <cuda.h>

#include <stdio.h>
#include <mutex>
#include <thread>
#include <vector>

#if ENABLE_PIX_RUNTIME
#include <Windows.h>
#include <pix3.h>
#include <ctime>
#endif

#define K_NOOP(...) (void)sizeof(0, __VA_ARGS__)

#if ENABLE_PIX_RUNTIME
#define K_PIX_BEGIN_EVENT(colorARGB, str) PIXBeginEvent(colorARGB, str)
#define K_PIX_END_EVENT()                 PIXEndEvent()
#else
#define K_PIX_BEGIN_EVENT(colorARGB, str) K_NOOP()
#define K_PIX_END_EVENT()                 K_NOOP()
#endif

#define NVRTC_SAFE_CALL(x)                                  \
	do {                                                    \
		nvrtcResult result = x;                             \
		if(result != NVRTC_SUCCESS) {                       \
			fprintf(stderr,                                 \
				"\nError: " #x " failed with error '%s'\n", \
				nvrtcGetErrorString(result));               \
			exit(1);                                        \
		}                                                   \
	} while(0)

#define CUDA_SAFE_CALL(x)                                   \
	do {                                                    \
		CUresult result = x;                                \
		if(result != CUDA_SUCCESS) {                        \
			const char *msg;                                \
			cuGetErrorName(result, &msg);                   \
			fprintf(stderr,                                 \
				"\nError: " #x " failed with error '%s'\n", \
				msg);                                       \
			exit(1);                                        \
		}                                                   \
	} while(0)

// https://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/1-Mandelbrot/Mandelbrot.html
const char * mandelbrot = R"(
__device__ unsigned int mandel_double(double cr, double ci, int max_iter)
{
	double zr = 0;
	double zi = 0;
	double zrsqr = 0;
	double zisqr = 0;

	unsigned int i;

	#pragma unroll // Note: used to increase compilation time
	for (i = 0; i < max_iter; i++){
		zi = zr * zi;
		zi += zi;
		zi += ci;
		zr = zrsqr - zisqr + cr;
		zrsqr = zr * zr;
		zisqr = zi * zi;
		
	//the fewer iterations it takes to diverge, the farther from the set
		if (zrsqr + zisqr > 4.0) break;
	}
	
	return i;
}

template <int MaxIter>
__global__ void mandel_kernel(unsigned int *counts, double xmin, double ymin,
			double step, int dim, unsigned int *colors)
{
	const int max_iter = MaxIter; 
	int pix_per_thread = dim * dim / (gridDim.x * blockDim.x);
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = pix_per_thread * tId;
	for (int i = offset; i < offset + pix_per_thread; i++){
		int x = i % dim;
		int y = i / dim;
		double cr = xmin + x * step;
		double ci = ymin + y * step;
		counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)];
	}
	if (gridDim.x * blockDim.x * pix_per_thread < dim * dim
			&& tId < (dim * dim) - (blockDim.x * gridDim.x)){
		int i = blockDim.x * gridDim.x * pix_per_thread + tId;
		int x = i % dim;
		int y = i / dim;
		double cr = xmin + x * step;
		double ci = ymin + y * step;
		counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)];
	}
	
})";


void CompileMandelbrotKernel(int maxIter)
{
	K_PIX_BEGIN_EVENT(0xFFFF5500, "CompileMandelbrotKernel");

	K_PIX_BEGIN_EVENT(0xFF88FF00, "nvrtcCreateProgram");
	nvrtcProgram prog;
	char programName[64];
	snprintf(programName, sizeof(programName), "mandelbrot_%d.cu", maxIter);
	NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, mandelbrot, programName, 0, nullptr, nullptr));
	K_PIX_END_EVENT();

	K_PIX_BEGIN_EVENT(0xFF0088FF, "nvrtcCompileProgram");

	const char * opts[] = { "--gpu-architecture=sm_80", "--std=c++14" };

	char entrypoint[64];
	snprintf(entrypoint, sizeof(entrypoint), "mandel_kernel<%d>", maxIter);
	NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, entrypoint));

	const nvrtcResult compileResult = nvrtcCompileProgram(prog, sizeof(opts)/sizeof(opts[0]), opts);

	K_PIX_END_EVENT();

	size_t logSize;
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	if(logSize > 1)
	{
		char * log = new char[logSize];
		NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
		fprintf(stderr, "%s\n", log);
		delete[] log;
	}

	if(compileResult != NVRTC_SUCCESS)
		exit(1);
  
	size_t ptxSize;
	NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
	char * ptx = new char[ptxSize];
	NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

	CUmodule module;
	CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
	delete[] ptx;

	const char * mangledEntrypoint = nullptr;
	NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, entrypoint, &mangledEntrypoint));
	CUfunction kernel;
	CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, mangledEntrypoint));

	NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
	CUDA_SAFE_CALL(cuModuleUnload(module));
	K_PIX_END_EVENT();
}

int main()
{
	CUdevice cuDevice;
	CUcontext context;
	CUDA_SAFE_CALL(cuInit(0));
	CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
	CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

	std::mutex mutex;
	std::condition_variable cv;
	bool bReady = false;

	const int numThreads = std::thread::hardware_concurrency() - 2;
	std::vector<std::thread> workers(numThreads);
	for(int i = 0; i < numThreads; ++i)
	{
		workers[i] = std::thread([&, i]()
		{
			CUDA_SAFE_CALL(cuCtxSetCurrent(context));

			std::unique_lock<std::mutex> lock(mutex);
			cv.wait(lock, [&]{ return bReady; });
			lock.unlock();

			CompileMandelbrotKernel(16 * (numThreads - i));
		});
	}

	#if ENABLE_PIX_RUNTIME
	HMODULE pixModule = PIXLoadLatestWinPixTimingCapturerLibrary();
	if(!pixModule)
	{
		DWORD errorCode = GetLastError();
		fprintf(stderr, "PIXLoadLatestWinPixTimingCapturerLibrary failed with error code: %d\n", errorCode);
		return 1;
	}

	wchar_t captureFilename[256];
	std::time_t time = std::time(nullptr);
	std::tm * calendarTime = std::localtime(&time);
	swprintf(captureFilename, sizeof(captureFilename)/sizeof(captureFilename[0]),
		L"NVRTC_Parallelism(%d-%02d-%02d.%02d-%02d-%02d).wpix",
		calendarTime->tm_year + 1900,
		calendarTime->tm_mon + 1,
		calendarTime->tm_mday,
		calendarTime->tm_hour,
		calendarTime->tm_min,
		calendarTime->tm_sec
	);

	PIXCaptureParameters params;
	params.TimingCaptureParameters.FileName = captureFilename;
	params.TimingCaptureParameters.CaptureCallstacks = true;
	params.TimingCaptureParameters.CaptureCpuSamples = 8000u; // must be 1000u, 4000, or 8000u. It's otherwise ignored.
	params.TimingCaptureParameters.CaptureFileIO = true;
	params.TimingCaptureParameters.CaptureVirtualAllocEvents = true;
	params.TimingCaptureParameters.CaptureHeapAllocEvents = true;
	params.TimingCaptureParameters.CaptureStorage = PIXCaptureParameters::PIXCaptureStorage::Memory;

	PIXBeginCapture(PIX_CAPTURE_TIMING, &params);
	#endif

	K_PIX_BEGIN_EVENT(0xFFFFFFFF, "main");

	{
		std::lock_guard<std::mutex> lock(mutex);
		bReady = true;
		cv.notify_all();
	}

	for(std::thread & w : workers)
		w.join();

	K_PIX_END_EVENT();

	#if ENABLE_PIX_RUNTIME
	PIXEndCapture(false);

	if(!FreeLibrary(pixModule))
		fprintf(stderr, "Failed to free PIX library\n");
	#endif

	CUDA_SAFE_CALL(cuCtxDestroy(context));

	return 0;
}