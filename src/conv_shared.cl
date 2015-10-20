__kernel void conv_shared(__global float * A, __constant float * B, __global float * C, int N, int M)
{
    int x_local_id = get_local_id(0);
    int y_local_id = get_local_id(1);

	int x_group_id = get_group_id(0);
	int y_group_id = get_group_id(1);


	__local localA[];
	__local localB[];
	
	if ((x_id < N) && (y_id < N))
	{
	  float val = 0;
	  for (int i = 0; i < M; ++i) {
		for (int j = 0; j < M; ++j) {
			val += A[(x_id - M / 2 + i) * N + (y_id - M / 2 + j)] * B[i * M + j];
		}
	  }
	  C[x_id * N + y_id] = val;
	}
}