__kernel void matrix_mult(__global float * A, __global float * B, __global float * C, int M, int N, int K)
{
    int x_id = get_global_id(0);
    int y_id = get_global_id(1);
	
	if ((x_id < M) && (y_id < K))
	{
	  float val = 0;
	  for (int i = 0; i < N; ++i) {
		  val += A[x_id * N + i] * B[i * K + y_id];
	  }
	  C[x_id * K + y_id] = val;
	}
}