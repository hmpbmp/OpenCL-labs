## OpenCL/CUDA laboratory

#### In this repository I keep source code for my OpenCL laboratories
[First laboratory task](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxwcmltYXRjdWRhfGd4OjRhZmZiN2QzMzBlNzVhN2U)


## Build

I use CMake as building tool to work with different IDEs and compilers.
My CMake version is 3.2.3 but there is no reason for errors when different version is used.
For example, build commands for Visual Studio 2015

```
1. mkdir build
2. cd build
3. cmake -G "Visual Studio 14 2015" ..
```
After this commands you can find `build` folder with VS 2015 solution.

## Input data

Input is generated using Python script `generate_data.py`. It is nessesary to have `build` folder already created.
It creates `input.txt` file in working directory and `py_output.txt` to compare with program result.
Example of usage:
```
python generate_data.py 1024 512 2048
```
## How to use
Program takes two command line arguments: input and output files.
```
<executable> <input file> <output file>
```
