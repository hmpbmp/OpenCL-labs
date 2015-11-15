__kernel void conv_global(__global float * A, __constant float * B, __global float * C, int N, int M)
{
    int x_id = get_global_id(0);
    int y_id = get_global_id(1);

	const int hfs = M / 2;
    const int dhfs = 2 * hfs;

	const int gl_row = N + M - 1;
	
	if ((hfs <= x_id) && (x_id < gl_row - hfs) && (hfs <= y_id) && (y_id < gl_row - hfs))
	{
	  float val = 0;
	  int t = 0;
	  for (int i = -hfs; i <= hfs; ++i) {
		for (int j = -hfs; j <= hfs; ++j, ++t) {
			val += A[(x_id + i) * gl_row + (y_id + j)] * B[t];
		}
	  }
	  C[x_id * gl_row + y_id] = val;
	}
}


__kernel void conv_shared(__global float * A, __constant float * B, __global float * C, __local float * localA, int N, int M)
{
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);
  
  const int x_local = get_local_id(0);
  const int y_local = get_local_id(1);

  const int hfs = M / 2;
  const int dhfs = 2 * hfs;

  const int local_dim = get_local_size(0) + dhfs;
  const int global_dim = N + M - 1;

  bool legal = (x_global >= hfs) && (y_global >= hfs) && (x_global < global_dim - hfs) && (y_global < global_dim - hfs);

  const int local_pos  = (x_local + hfs) * local_dim  + (y_local + hfs);
  const int global_pos = (x_global)      * global_dim + y_global;

  localA[local_pos] = A[global_pos];
	
	int x_shift = 0;

	if (x_local < hfs) {
		if (x_global >= hfs) {
		  localA[local_pos - hfs * local_dim] = A[global_pos - hfs * global_dim];
		  x_shift = -hfs;
		}
	}

	if (x_local >= get_local_size(0) - hfs) {
		if (x_global < global_dim - hfs) {
		  localA[local_pos + hfs * local_dim] = A[global_pos + hfs * global_dim];
		  x_shift = hfs;
		}
	}

	if (y_local < hfs) {
		if (y_global >= hfs) {
		  localA[local_pos - hfs] = A[global_pos - hfs];
		  if (x_shift) {
			  localA[local_pos - hfs + x_shift * local_dim] = A[global_pos - hfs + x_shift * global_dim];
		  }
	    }
	}


	if (y_local >= get_local_size(0) - hfs) {
		if (y_global < global_dim - hfs) {
		  localA[local_pos + hfs] = A[global_pos + hfs];
		  if (x_shift) {
		      localA[local_pos + hfs + x_shift * local_dim] = A[global_pos + hfs + x_shift * global_dim];
		  }
	  }
	}

  barrier(CLK_LOCAL_MEM_FENCE);
	
	
  if(legal) {
  
  float val = 0.0f;
	int t = 0;
	for (int i = -hfs; i <= hfs; ++i) {
		for (int j = -hfs; j <= hfs; ++j,++t) {
			float k = localA[(x_local + hfs + i) * local_dim + (y_local + hfs + j)];
			val += localA[(x_local + hfs + i) * local_dim + (y_local + hfs + j)] * B[t];
		}
	}
	
	C[global_pos] = val;

  }

}