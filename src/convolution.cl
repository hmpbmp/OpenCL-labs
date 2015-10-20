__kernel void conv_global(__global float * A, __constant float * B, __global float * C, int N, int M)
{
    int x_id = get_global_id(0);
    int y_id = get_global_id(1);
	
	if ((M / 2 < x_id && x_id < N) && (M / 2 < y_id && y_id < N))
	{
	  float val = 0;
	  for (int i = 0; i < M; ++i) {
		for (int j = 0; j < M; ++j) {
			val += A[(x_id - M / 2 + i) * (N + M) + (y_id - M / 2 + j)] * B[i * M + j];
		}
	  }
	  C[x_id * (N + M) + y_id] = val;
	}
}


__kernel void conv_shared(__global float * A, __constant float * B, __global float * C, int N, int M)
{
    int x_local_id = get_local_id(0);
    int y_local_id = get_local_id(1);

	int x_group_id = get_group_id(0);
	int y_group_id = get_group_id(1);

	int x_id = x_group_id * M + x_local_id;
	int y_id = y_group_id * M + y_local_id;

	__local localA[M * M];

	localA[x_local_id * M + y_local_id] = A[x_id * (N + M) + y_id];

	barrier(CLK_LOCAL_MEM_FENCE);
	
	if ((M / 2 < x_id && x_id < N) && (M /2 < y_id && y_id < N))
	{
	  float val = 0;
	  for (int i = 0; i < M; ++i) {
		for (int j = 0; j < M; ++j) {
			val += localA[i * M + j] * B[i * M + j];
		}
	  }
	  C[x_id * (N + M) + y_id] = val;
	}
}