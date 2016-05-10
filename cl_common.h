#ifndef CL_COMMON_H_
#define CL_COMMON_H_

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include <stdio.h>

struct cl_device_properties {
	char name[256];

	cl_ulong global_mem_size;
	cl_ulong max_mem_alloc;

	size_t max_work_group_size;
};

struct cl_state {
	cl_int		error;

	cl_uint		plat_cnt;
	cl_platform_id	*platforms;
	cl_uint enabled_platforms;

	cl_context	context;
	cl_program	program;

	cl_uint		dev_cnt;
	cl_device_id	*devices;
	struct cl_device_properties *dev_props;

	cl_kernel	*kernels;
	cl_command_queue *queues;
	cl_event	*events;
};

void destroy_cl_state(struct cl_state *cl);

cl_int populate_platforms(struct cl_state *cl);

cl_int populate_devices(struct cl_state *cl);

cl_int create_context(struct cl_state *cl);

cl_int create_queues(struct cl_state *cl);

cl_int build_program(struct cl_state *cl, char *fname);

cl_int create_kernels(struct cl_state *cl, char *kname);

const char *cl_errno_str(cl_int error);

#endif
