## OpenCL/CUDA laboratory

#### In this repository I keep source code for my OpenCL laboratories
[Second laboratory task](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxwcmltYXRjdWRhfGd4OjczZWE5MmFkZjk1N2EyZGI)


## Build

I use CMake as building tool to work with different IDEs and compilers.
My CMake version is 3.2.3 but there is no reason for errors when different version is used.
For example, build commands for Visual Studio 2015 (out-of-source build)

```
1. mkdir build
2. cd build
3. cmake -G "Visual Studio 14 2015 Win64" ..
```
After this commands you can find `build` folder with VS 2015 solution.

## Input data

## How to use
Program takes two command line arguments: input and output files.
```
<executable> <input file> <output file>
```
