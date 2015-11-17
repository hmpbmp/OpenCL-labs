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

#define OPTIMIZATION

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
    std::ifstream cl_file("../src/reduction.cl");
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
    int N;
    input >> N;

    float *A = new float [N];

    for (int i = 0; i < N; ++i) {
      input >> A[i];
    }

    input.close();

    int limit = max_wg_size;
    const size_t LOCAL_DIM = limit;
    float *B;
    cl_ulong sum_time = 0;

    while (N >= 2 * limit) {
      int size_A = sizeof(float) * N;
      cl::Buffer dev_A(context, CL_MEM_READ_ONLY, size_A);
      queue.enqueueWriteBuffer(dev_A, CL_TRUE, 0, size_A, A);
      size_t GLOBAL_DIM = N % limit == 0 ? N : N + limit - N % limit;

      int M = GLOBAL_DIM / LOCAL_DIM;
      B = new float[M];
      int size_B = sizeof(float) * M;

      cl::Buffer dev_B(context, CL_MEM_READ_WRITE, size_B);
      queue.enqueueWriteBuffer(dev_B, CL_TRUE, 0, size_B, B);

      //Load kernel from OpenCL source
#undef OPTIMIZATION
#ifndef OPTIMIZATION
      printf("Using simple reduction\n");
      const size_t local_size = limit * sizeof(float);
      auto reduction = cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::LocalSpaceArg&, int>(program, "reduction");
      cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(GLOBAL_DIM), cl::NDRange(LOCAL_DIM));
      cl::Event evt = reduction(eargs, dev_A, dev_B, cl::__local(local_size), N);
#else
      printf("Using optimized reduction\n");
      const size_t local_size = limit * sizeof(float);
      auto opt_reduction = cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::LocalSpaceArg&, int>(program, "opt_reduction");
      cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(GLOBAL_DIM), cl::NDRange(LOCAL_DIM));
      cl::Event evt = opt_reduction(eargs, dev_A, dev_B, cl::__local(local_size), N);
#endif
      evt.wait();
      cl_ulong start_time = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end_time = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      cl_ulong time = end_time - start_time;
      sum_time += time;
      //Copy from device to host
      queue.enqueueReadBuffer(dev_B, CL_TRUE, 0, size_B, B);
      delete[] A;
      A = B;
      N = M;
    }

    FILE *f = std::fopen(argv[2], "w+");
    if (f == NULL) {
      throw std::ios_base::failure(NULL);
    }
    for (int i = 1; i < N; ++i) {
      B[0] += B[i];
    }
    fprintf(f, "%.2f ", B[0]);
    fclose(f);
    delete[] B;
    std::cout << std::setprecision(3) << "Total time: " << sum_time / (float)10e6 << " ms" << std::endl;

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