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
    int K = N + M;

    float *A = new float [K * K];
    float *B = new float [M * M];
    float *C = new float [K * K];

    for (int j = 0; j < K; ++j)
    {
      A[j] = 0;
      A[(K - 1) * K + j] = 0;
    }

    for (int i =  M / 2; i < K - M / 2; ++i) {
      for (int j = 0; j < M / 2; ++j) {
        A[K * i + j] = 0;
      }
      for (int j = M / 2; j < K - M / 2; ++j) {
        input >> A[K * i + j];
      }
      for (int j = K - M / 2; j < K; ++j) {
        A[K * i + j] = 0;
      }
    }

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < M; ++j) {
        input >> B[M * i + j];
      }
    }

    input.close();

    //Allocate device buffers for data
    int size_A = sizeof(float) * (N + M) * (N + M);
    int size_B = sizeof(float) * M * M;
    int size_C = sizeof(float) * (N + M) * (N + M);

    cl::Buffer dev_A(context, CL_MEM_READ_ONLY, size_A);
    cl::Buffer dev_B(context, CL_MEM_READ_ONLY, size_B);
    cl::Buffer dev_C(context, CL_MEM_WRITE_ONLY, size_C);

    //Copy from host to command queue
    queue.enqueueWriteBuffer(dev_A, CL_TRUE, 0, size_A, A);
    queue.enqueueWriteBuffer(dev_B, CL_TRUE, 0, size_B, B);

    
    //Define work-group and work-item sizes
    int limit = (int)sqrt(max_wg_size);


    const size_t X_LOCAL_DIM = M;//(int)(sqrt(max_wg_size));
    const size_t Y_LOCAL_DIM = M;//(int)(sqrt(max_wg_size));
    const size_t X_GLOBAL_DIM = 0;// d1;
    const size_t Y_GLOBAL_DIM = 0;//;


    //Load kernel from OpenCL source
    auto conv_global = cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer&, int , int, int>(program, "conv_global");
    cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(X_GLOBAL_DIM, Y_GLOBAL_DIM), cl::NDRange(X_LOCAL_DIM, Y_LOCAL_DIM));
    //matrix_mult(eargs, dev_A, dev_B, dev_C, M, N, K).wait();

    //Copy from device to host
    queue.enqueueReadBuffer(dev_C, CL_TRUE, 0, size_C, C);

    FILE *f = std::fopen(argv[2], "w+");
    if (f == NULL) {
      throw std::ios_base::failure(NULL);
    }
    for (int i = M / 2; i < K - M / 2; ++i) {
      for (int j = M / 2; j < K - M / 2; ++j) {
        fprintf(f, "%.2f ", C[i * K + j]);
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