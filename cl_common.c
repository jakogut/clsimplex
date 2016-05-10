#include "cl_common.h"

#include <stdio.h>
#include <stdlib.h>

void destroy_cl_state(struct cl_state *cl)
{
	if (cl->events)
		free(cl->events);
	if (cl->queues)
		free(cl->queues);
	if (cl->kernels)
		free(cl->kernels);
	if (cl->dev_props)
		free(cl->dev_props);
	if (cl->devices)
		free(cl->devices);
	if (cl->platforms)
		free(cl->platforms);
}

static cl_uint get_plat_cnt(void)
{
	cl_uint plat_cnt;

	clGetPlatformIDs(0, NULL, &plat_cnt);
	return plat_cnt;
}

cl_int populate_platforms(struct cl_state *cl)
{
	cl->plat_cnt = get_plat_cnt();

	if (!cl->plat_cnt) {
		printf("ERROR: %s\n", cl_errno_str(cl->error));
		return -1;
	}

	cl->platforms = calloc(cl->plat_cnt, sizeof(cl_platform_id));
	cl->error = clGetPlatformIDs(cl->plat_cnt, cl->platforms, NULL);

	if (cl->error != CL_SUCCESS) {
		printf("ERROR: %s\n", cl_errno_str(cl->error));
		return cl->error;
	}

	return cl->error;
}

static cl_uint get_dev_cnt(cl_device_type dt, cl_platform_id p)
{
	cl_uint dev_cnt;

	clGetDeviceIDs(p, dt, 0, NULL, &dev_cnt);
	return dev_cnt;
}

cl_int populate_devices(struct cl_state *cl)
{
	cl->dev_cnt = get_dev_cnt(CL_DEVICE_TYPE_ALL, cl->platforms[0]);

	if (!cl->dev_cnt)
		return cl->error;

	cl->devices = calloc(cl->dev_cnt, sizeof(cl_device_id));

	cl->error = clGetDeviceIDs(cl->platforms[0],
			   CL_DEVICE_TYPE_ALL,
			   cl->dev_cnt,
			   cl->devices,
			   NULL);

	if (cl->error != CL_SUCCESS) {
		printf("ERROR: %s\n", cl_errno_str(cl->error));
		return cl->error;
	}

	cl->dev_props = calloc(cl->dev_cnt,
				sizeof(struct cl_device_properties));

	for (unsigned i = 0; i < cl->dev_cnt; i++) {
		clGetDeviceInfo(cl->devices[i],
			CL_DEVICE_NAME, 256,
			cl->dev_props[i].name, NULL);

		clGetDeviceInfo(cl->devices[i],
			CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(size_t),
			&cl->dev_props[i].max_work_group_size,
			NULL);
	}

	return cl->error;
}

cl_int create_context(struct cl_state *cl)
{
	cl->context = clCreateContext(NULL,
			      cl->dev_cnt,
			      cl->devices,
			      NULL, NULL,
			      &cl->error);

	if (cl->error != CL_SUCCESS) {
		printf("ERROR: Failed to create context with %s\n",
			cl_errno_str(cl->error));

		return cl->error;
	}

	return cl->error;
}

cl_int create_queues(struct cl_state *cl)
{
	cl->queues = calloc(cl->dev_cnt, sizeof(cl_command_queue));

	for (unsigned i = 0; i < cl->dev_cnt; i++) {
		cl->queues[i] = clCreateCommandQueue(cl->context,
				     cl->devices[i],
				     CL_QUEUE_PROFILING_ENABLE,
				     &cl->error);

		if (cl->error != CL_SUCCESS) {
			printf("Failed to create command queue with error %s\n",
			       cl_errno_str(cl->error));

			return cl->error;
		}
	}

	cl->events = calloc(cl->dev_cnt, sizeof(cl_event));
	return cl->error;
}

static int cl_fcopy(char *dest, size_t size, FILE *src)
{
	size_t i;
	long pos = ftell(src);

	for (i = 0; i < size && !feof(src); i++)
		dest[i] = fgetc(src);
	fseek(src, pos, SEEK_SET);

	return 0;
}

static int cl_flength(FILE *f)
{
	int length;
	long pos = ftell(f);

	for (length = 0; !feof(f); length++)
		fgetc(f);

	fseek(f, pos, SEEK_SET);

	return length - 1;
}

cl_int get_program_build_info(struct cl_state *cl) {
    size_t log_size;
    cl_int err = 0;
    char *build_log;

    for (cl_uint dev = 0; dev < cl->dev_cnt; dev++) {
        err = clGetProgramBuildInfo(cl->program, cl->devices[dev], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        build_log = (char* )malloc((log_size+1));

        // Second call to get the log
        err = clGetProgramBuildInfo(cl->program, cl->devices[dev], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf("--- Build log ---\n ");
        fprintf(stderr, "%s\n", build_log);
        free(build_log);
    }

    return err;
}

cl_int build_program(struct cl_state *cl, char *fname)
{
	FILE *f = fopen(fname, "rb");
	size_t src_size = cl_flength(f);
	cl_uint num_src_files = 1;

	char *source = calloc(src_size, sizeof(char));

	cl_fcopy(source, src_size, f);
	fclose(f);

	cl->program = clCreateProgramWithSource(cl->context,
					num_src_files,
					(const char **)&source,
					&src_size, &cl->error);

	if (cl->error != CL_SUCCESS) {
		printf("ERROR: Failed to create program with %s\n",
			cl_errno_str(cl->error));

		return cl->error;
	}

	free(source);

	cl->error = clBuildProgram(cl->program, cl->dev_cnt, cl->devices,
				NULL, NULL, NULL);

	if (cl->error != CL_SUCCESS) {
		printf("ERROR: Failed building program with %s\n",
			cl_errno_str(cl->error));

        get_program_build_info(cl);
	}

	return cl->error;
}

cl_int create_kernels(struct cl_state *cl, char *kname)
{
	cl->kernels = calloc(cl->dev_cnt, sizeof(cl_kernel));

	for (unsigned i = 0; i < cl->dev_cnt; i++) {
		cl->kernels[i] = clCreateKernel(cl->program, kname, &cl->error);

		if (cl->error != CL_SUCCESS)
			printf("ERROR: Failed to create kernel.\n");
	}

	return cl->error;
}

const char *cl_errno_str(cl_int err)
{
	switch (err) {
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";

	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	default:  return "Unknown OpenCL error";
	}
}
