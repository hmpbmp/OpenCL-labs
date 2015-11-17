__kernel void reduction(__global float * A, __global float * B, __local float * localA, int N)
{
  size_t id_global = get_global_id(0);
  size_t id_local = get_local_id(0);
  size_t local_size = get_local_size(0);

  localA[id_local] = A[id_global];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (size_t s = 1; s < local_size; s*=2) {
	int index = 2 * s * id_local;
	if (index < local_size) {
		localA[index] += localA[index + s];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (id_local == 0) {
	B[get_group_id(0)] = localA[0];
  }

}


__kernel void opt_reduction(__global float * A, __global float * B, __local float * localA, int N)
{
  size_t id_global = get_global_id(0);
  size_t id_local = get_local_id(0);
  size_t local_size = get_local_size(0);

  int i = get_group_id(0) * (2 * local_size) + id_local;
  int grSize = get_group_id(0) * 2 * get_num_groups(0);
  localA[id_local] = A[id_global];
  barrier(CLK_LOCAL_MEM_FENCE);



  if (local_size >= 1024) {
	if (id_local < 512) {
		localA[id_local] += localA[id_local + 512];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_size >= 512) {
	if (id_local < 256) {
		localA[id_local] += localA[id_local + 256];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_size >= 256) {
	if (id_local < 128) {
		localA[id_local] += localA[id_local + 128];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_size >= 128) {
	if (id_local < 64) {
		localA[id_local] += localA[id_local + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (id_local < 32) {
	if (local_size >= 64) { localA[id_local] += localA[id_local + 32]; }
	if (local_size >= 32) { localA[id_local] += localA[id_local + 16]; }
	if (local_size >= 16) { localA[id_local] += localA[id_local +  8]; }
	if (local_size >=  8) { localA[id_local] += localA[id_local +  4]; }
	if (local_size >=  4) { localA[id_local] += localA[id_local +  2]; }
	if (local_size >=  2) { localA[id_local] += localA[id_local +  1]; }
  }

  if (id_local == 0) {
	B[get_group_id(0)] = localA[0];
  }

}
