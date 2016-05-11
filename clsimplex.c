#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include <CL/cl.h>
#include "cl_common.h"

#define XS 256
#define YS 256
#define ZS 256

#define CHUNK_SIZE XS * YS * ZS

void print_chunk(float *chunk, unsigned xs, unsigned ys, unsigned zs) {
    for (unsigned z = 0; z < zs; z++) {
        //printf("\n\n");

        for (unsigned x = 0; x < xs; x++) {
            printf("\n");

            for (unsigned y = 0; y < ys; y++) {
                float val = chunk[x + (y*ys) + (z*zs*zs)];
                if (val > 0) printf(" . ");
                else printf(" # ");
            }
        }
    }

    printf("\n\n");
}

void shuffle_uint8(uint8_t *array, size_t n)
{
    size_t i;
    for (i = n - 1; i > 0; i--) {
        size_t j = rand() % (i+1);
        uint8_t t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

unsigned char *compute_seed()
{
    unsigned char *perm = calloc(512, sizeof(unsigned char));
    for (int i = 0; i < 256; i++) perm[i] = i;
    shuffle_uint8(perm, 256);
    for(int i = 0; i < 256; i++) perm[i + 256] = perm[i];

    return perm;
}

int main(void)
{
	struct cl_state cl;

	populate_platforms(&cl);
	populate_devices(&cl);
	create_context(&cl);
	create_queues(&cl);
	build_program(&cl, "simplex.cl");
	create_kernels(&cl, "sdnoise3");

	cl_mem *res_d = calloc(cl.dev_cnt, sizeof(cl_mem));

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		res_d[i] = clCreateBuffer(cl.context,
                      CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY,
					  CHUNK_SIZE * sizeof(float),
					  NULL, &cl.error);
	}

	srand(time(NULL));
    unsigned char *perm = compute_seed();

    cl_mem perm_d = clCreateBuffer(cl.context,
                    CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                    sizeof(uint8_t)*512,
                    perm, &cl.error);

	if (cl.error != CL_SUCCESS)
		printf("Failed to create device buffers with %s\n",
			cl_errno_str(cl.error));

    const float x=0, y=0, z=0;
    const unsigned xs=XS, ys=YS, zs=ZS;

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		cl.error = clSetKernelArg(cl.kernels[i], 0,
					  sizeof(float), &x);
		cl.error |= clSetKernelArg(cl.kernels[i], 1,
					   sizeof(float), &y);
		cl.error |= clSetKernelArg(cl.kernels[i], 2,
					   sizeof(float), &z);
		cl.error |= clSetKernelArg(cl.kernels[i], 3,
					   sizeof(unsigned), &xs);
		cl.error |= clSetKernelArg(cl.kernels[i], 4,
					   sizeof(unsigned), &ys);
		cl.error |= clSetKernelArg(cl.kernels[i], 5,
					   sizeof(unsigned), &zs);
		cl.error |= clSetKernelArg(cl.kernels[i], 6,
					   sizeof(cl_mem), &res_d[i]);
        cl.error |= clSetKernelArg(cl.kernels[i], 7,
                       sizeof(cl_mem), &perm_d);

		if (cl.error != CL_SUCCESS)
			printf("Error while settings kernel args: %s\n",
			       cl_errno_str(cl.error));
	}

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		const size_t local_ws = cl.dev_props[i].max_work_group_size;
		const size_t global_ws = CHUNK_SIZE + (CHUNK_SIZE % local_ws);

		cl.error = clEnqueueNDRangeKernel(cl.queues[i],
						  cl.kernels[i],
						  1,
						  0,
						  &global_ws,
						  &local_ws,
						  0,
						  NULL,
						  &cl.events[i]);
		if (cl.error != CL_SUCCESS)
			printf("ERROR: Kernel failed to run on GPU. %s\n",
			       cl_errno_str(cl.error));
	}

	float *device_result = calloc(CHUNK_SIZE, sizeof(float));

	for (unsigned i = 0; i < cl.dev_cnt; i++) {
		clWaitForEvents(1, &cl.events[i]);

		cl_ulong time_start = 0, time_end = 0;

		clFinish(cl.queues[i]);
		cl.error = clWaitForEvents(1, &cl.events[i]);

		cl.error = clGetEventProfilingInfo(cl.events[i],
					CL_PROFILING_COMMAND_START,
					sizeof(cl_ulong), &time_start, NULL);

		cl.error = clGetEventProfilingInfo(cl.events[i],
					CL_PROFILING_COMMAND_END,
					sizeof(cl_ulong), &time_end, NULL);

		cl.error = clEnqueueReadBuffer(cl.queues[i],
					       res_d[i],
					       CL_TRUE,
					       0,
					       CHUNK_SIZE * sizeof(float),
					       device_result,
					       0,
					       NULL,
					       NULL);

		if (cl.error != CL_SUCCESS)
			printf("Copy device buffer to host failed with %s",
			       cl_errno_str(cl.error));

		printf("GPU %i: %s\n", i, cl.dev_props[i].name);

		double sec_elapsed_gpu = (time_end - time_start)
					 / 1000000000.0f;

        printf("Elapsed time: %f\n\n", sec_elapsed_gpu);
	}

    print_chunk(device_result, 32, 32, 32);

	free(res_d);
	free(device_result);

	destroy_cl_state(&cl);

	return 0;
}
