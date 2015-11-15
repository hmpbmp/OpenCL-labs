#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <utility>
#include <string>
#include <cmath>

#define USE_LOCAL_MEMORY

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "Unexpected arguments number." << std::endl;
    std::cout << "Programm usage: lab1 <input-file> <output-file>" << std::endl;
    return -1;
  }

  try {
    std::ifstream input(argv[1]);
    input.exceptions(std::ifstream::failbit);


	  std::vector<cl::Platform> platforms;
	  std::vector<cl::Device> devices;
	  std::vector<cl::Kernel> kernels;


    //Create platform and device layers
    cl::Platform::get(&platforms);
    std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    //Device info
    std::string name;
    devices[0].getInfo<std::string>(CL_DEVICE_NAME, &name);
    std::cout << "Using device: " << name  << std::endl;
    size_t max_wg_size;
    devices[0].getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_wg_size);
    std::cout << "MAX_WORK_GROUP_SIZE: " << max_wg_size << std::endl;
    std::vector<size_t> max_wi_sizes;
    devices[0].getInfo<std::vector<size_t>>(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_wi_sizes);
    std::cout << "MAX_WORK_ITEM_SIZES: ";
    for (auto i : max_wi_sizes) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
    size_t max_cmp_units;
    devices[0].getInfo<size_t>(CL_DEVICE_MAX_COMPUTE_UNITS, &max_cmp_units);
    std::cout << "MAX_WORK_COMPUTE_UNITS: " << max_cmp_units << std::endl;

    //Create context and command queue
    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    //Load OpenCL source 
    std::ifstream cl_file("../src/convolution.cl");
    cl_file.exceptions(std::ifstream::failbit);

    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source;
    source.push_back(std::make_pair(cl_string.c_str(), cl_string.length() + 1));

    //Create and compile program
    cl::Program program(context, source);
    try {
      program.build(devices);
    }
    catch(cl::Error const &buildErr) {
      std::cout << std::endl << "Error type:" << buildErr.what() << std::endl;
      std::cout << "Error code" << buildErr.err() << std::endl;
      std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      return -1;
    }

    //Create data to send to kernel
    int M, N;
    input >> N >> M;

    float *A = new float [(N + M - 1) * (N + M - 1)];
    float *B = new float [M * M];
    float *C = new float [(N + M - 1) * (N + M - 1)];

    for (int i = 0; i < M / 2; ++i) {
      for (int j = 0; j < (N + M - 1); ++j) {
        A[i * (N + M - 1) + j] = 0;
      }
    }

    for (int i =  M / 2; i < N + M / 2; ++i) {
      for (int j = 0; j < M / 2; ++j) {
        A[(N + M - 1) * i + j] = 0;
      }
      for (int j = M / 2; j < N + M / 2; ++j) {
        input >> A[(N + M - 1) * i + j];
      }
      for (int j = N + M / 2; j < (N + M - 1); ++j) {
        A[(N + M - 1) * i + j] = 0;
      }
    }

    for (int i = N + M / 2; i < (N + M - 1); ++i) {
      for (int j = 0; j < (N + M - 1); ++j) {
        A[(N + M - 1) * i + j] = 0;
      }
    }

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < M; ++j) {
        input >> B[M * i + j];
      }
    }

    input.close();

    //Allocate device buffers for data
    int size_A = sizeof(float) * (N + M - 1) * (N + M - 1);
    int size_B = sizeof(float) * M * M;
    int size_C = sizeof(float) * (N + M - 1) * (N + M - 1);

    cl::Buffer dev_A(context, CL_MEM_READ_ONLY, size_A);
    cl::Buffer dev_B(context, CL_MEM_READ_ONLY, size_B);
    cl::Buffer dev_C(context, CL_MEM_WRITE_ONLY, size_C);

        

    //Copy from host to command queue
    queue.enqueueWriteBuffer(dev_A, CL_TRUE, 0, size_A, A);
    queue.enqueueWriteBuffer(dev_B, CL_TRUE, 0, size_B, B);

    
    //Define work-group and work-item sizes
    int limit = (int)sqrt(max_wg_size);


    const size_t X_LOCAL_DIM = limit;
    const size_t Y_LOCAL_DIM = limit;
    const size_t X_GLOBAL_DIM = (N + M - 1) + limit - (N + M - 1) % limit;
    const size_t Y_GLOBAL_DIM = (N + M - 1) + limit - (N + M - 1) % limit;
    
    //cl::Buffer tmp(context, CL_MEM_WRITE_ONLY, (X_LOCAL_DIM + M / 2) * (Y_LOCAL_DIM + M / 2));
    

    //Load kernel from OpenCL source
//#undef USE_LOCAL_MEMORY

#ifdef USE_LOCAL_MEMORY
    printf("Using local memory\n");
    const size_t local_size = (limit + M - 1) * (limit + M - 1) * sizeof(float);
    //cl::Buffer tmp(context, CL_MEM_READ_WRITE, local_size);
    auto conv_shared = cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer&, cl::LocalSpaceArg& , int, int>(program, "conv_shared");
    cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(X_GLOBAL_DIM, Y_GLOBAL_DIM), cl::NDRange(X_LOCAL_DIM, Y_LOCAL_DIM));
    cl::Event evt = conv_shared(eargs, dev_A, dev_B, dev_C, cl::__local(local_size), N, M);
#else
    printf("Using global memory\n");
    auto conv_global = cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer&, int, int>(program, "conv_global");
    cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(X_GLOBAL_DIM, Y_GLOBAL_DIM), cl::NDRange(X_LOCAL_DIM, Y_LOCAL_DIM));
    cl::Event evt = conv_global(eargs, dev_A, dev_B, dev_C, N, M);
#endif

    evt.wait();
    cl_ulong start_time = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong time = end_time - start_time;
    std::cout << std::setprecision(3) << "Total time: " << time / (float)10e6 << " ms" << std::endl;
    //Copy from device to host
    queue.enqueueReadBuffer(dev_C, CL_TRUE, 0, size_C, C);

    FILE *f = std::fopen(argv[2], "w+");
    if (f == NULL) {
      throw std::ios_base::failure(NULL);
    }

    for (int i = M / 2; i < N + M / 2; ++i) {
      for (int j = M  / 2; j < N + M / 2; ++j) {
        fprintf(f, "%2.2f ", C[i * (N + M - 1) + j]);
      }
      fprintf(f, "\n");
    }
    fclose(f);
  }
  catch(cl::Error const & err){
    std::cout << std::endl << "Error type:" << err.what() << std::endl;
    std::cout << "Error code" << err.err() << std::endl;
  }
  catch (std::ios_base::failure &fail) {
    std::cout << "Failed to open file. Check file path." << std::endl;
  }
  catch (std::bad_array_new_length &err){
    std::cout << "Allocation error: std::bad_array_new_length" << std::endl;
  }

	return 0;
}