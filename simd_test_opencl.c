/*
 * Project : SIMD_Utils
 * Version : 0.1.12
 * Author  : JishinMaster
 * Licence : BSD-2
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

/** forces opencl version to 1.2 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "simd_utils.h"

#define CACHE_LINE_SIZE 64
#define MAX_PLATFORM 4
#define MAX_DEVICE 4
#define MAX_PLATFORM_NAME 128
#define MAX_DEVICE_NAME 128
#define MAX_EVENTS 6

enum alloc_type { NATIVE,
                  NATIVE_COPY,
                  ALLOC_HOST_PTR };

/* Find a GPU or CPU associated with the first available platform
The `platform` structure identifies the first platform identified by the
OpenCL runtime. A platform identifies a vendor's installation, so a system
may have an NVIDIA platform and an AMD platform.
The `device` structure corresponds to the first accessible device
associated with the platform. Because the second parameter is
`CL_DEVICE_TYPE_opencl`, this device must be a GPU.
*/
cl_device_id create_device(int platform_choice, int device_choice)
{
    cl_int err;
    cl_platform_id platform[MAX_PLATFORM];  // maximum 4 platform
    cl_uint num_platforms = 0, num_devices = 0;
    cl_device_id device[MAX_DEVICE];
    int ret = 0;
    char platform_name[MAX_PLATFORM_NAME];
    char device_name[MAX_DEVICE_NAME];

    size_t ret_param_size;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(MAX_PLATFORM, platform, &num_platforms);
    if (err < 0) {
        printf("Error clGetPlatformIDs %d, line %d\n", err, __LINE__);
        exit(1);
    }

    err =
        clGetPlatformInfo(platform[platform_choice], CL_PLATFORM_NAME,
                          sizeof(platform_name), platform_name, &ret_param_size);
    if (err < 0) {
        printf("Error clGetPlatformInfo %d, line %d\n", err, __LINE__);
        exit(1);
    }

    printf("Platform found: %s\n", platform_name);

    err = clGetDeviceIDs(platform[platform_choice], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICE, device, &num_devices);
    if (err < 0) {
        printf("Error clGetDeviceIDs %d,line %d\n", err, __LINE__);
        exit(1);
    }

    err = clGetDeviceInfo(device[device_choice], CL_DEVICE_NAME,
                          sizeof(device_name), device_name, &ret_param_size);
    if (err < 0) {
        printf("Error clGetDeviceInfo %d, line %d\n", err, __LINE__);
        exit(1);
    }

    printf("Device found on the above platform: %s\n", device_name);

    return device[device_choice];
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename, const char *options)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file
  Creates a program from the source code in the add_numbers.cl file.
  Specifically, the code reads the file's content into a char array
  called program_buffer, and then calls clCreateProgramWithSource.
  */
    program = clCreateProgramWithSource(ctx, 1, (const char **) &program_buffer,
                                        &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program
  The fourth parameter accepts options that configure the compilation.
  These are similar to the flags used by gcc. For example, you can
  define a macro with the option -DMACRO=VALUE and turn off optimization
  with -cl-opt-disable.
  */
    err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (err < 0) {
        printf("Error building program : \n");
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
                              &log_size);
        program_log = (char *) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
                              program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

int write_file(const char *name, const unsigned char *content, size_t size)
{
    FILE *fp = fopen(name, "wb+");
    if (!fp) {
        return -1;
    }
    fwrite(content, size, 1, fp);
    fclose(fp);
    return 0;
}

cl_int write_binaries(cl_program program, unsigned num_devices, int platform_choice)
{
    unsigned i;
    cl_int err = CL_SUCCESS;
    size_t *binaries_size = NULL;
    unsigned char **binaries_ptr = NULL;

    // Read the binaries size
    size_t binaries_size_alloc_size = sizeof(size_t) * num_devices;
    binaries_size = (size_t *) malloc(binaries_size_alloc_size);
    if (!binaries_size) {
        err = CL_OUT_OF_HOST_MEMORY;
        goto cleanup;
    }

    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                           binaries_size_alloc_size, binaries_size, NULL);
    if (err != CL_SUCCESS) {
        goto cleanup;
    }

    // Read the binaries
    size_t binaries_ptr_alloc_size = sizeof(unsigned char *) * num_devices;
    binaries_ptr = (unsigned char **) malloc(binaries_ptr_alloc_size);
    if (!binaries_ptr) {
        err = CL_OUT_OF_HOST_MEMORY;
        goto cleanup;
    }
    memset(binaries_ptr, 0, binaries_ptr_alloc_size);
    for (i = 0; i < num_devices; ++i) {
        binaries_ptr[i] = (unsigned char *) malloc(binaries_size[i]);
        if (!binaries_ptr[i]) {
            err = CL_OUT_OF_HOST_MEMORY;
            goto cleanup;
        }
    }

    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                           binaries_ptr_alloc_size,
                           binaries_ptr, NULL);
    if (err != CL_SUCCESS) {
        goto cleanup;
    }

    // Write the binaries to file
    for (i = 0; i < num_devices; ++i) {
        // Create output file name
        char filename[128];
        snprintf(filename, sizeof(filename), "cl-out_%u-%u.bin",
                 (unsigned) platform_choice, (unsigned) i);

        // Write the binary to the output file
        write_file(filename, binaries_ptr[i], binaries_size[i]);
    }

cleanup:
    // Free the return value buffer
    if (binaries_ptr) {
        for (i = 0; i < num_devices; ++i) {
            free(binaries_ptr[i]);
        }
        free(binaries_ptr);
    }
    free(binaries_size);

    return err;
}

size_t rounded_size_aligned(size_t size, size_t alignment)
{
    return size + (size) % alignment;
}

float l2_err(float *test, float *ref, int len)
{
    float l2_err = 0.0f;

    for (int i = 0; i < len; i++) {
        l2_err += (ref[i] - test[i]) * (ref[i] - test[i]);
    }

#ifdef RELEASE
    if (l2_err > 0.00001f)
        printf("L2 ERR %0.7f\n", l2_err);
#else
    printf("L2 ERR %0.7f\n", l2_err);
#endif
    return l2_err;
}

/** allocates aligned memory on host only if required, returns NULL otherwise */
cl_mem alloc_host(cl_context ctx, cl_command_queue queue, cl_map_flags flags, void **ptr, size_t size, int alloc_type)
{
    cl_mem host_mem;
    cl_int err;
    *ptr = NULL;

    if ((alloc_type == NATIVE) || (alloc_type == NATIVE_COPY)) {
        *ptr = aligned_alloc(CACHE_LINE_SIZE, size);
        if (*ptr == NULL) {
            printf("Error allocating device of size %d\n", size);
            exit(1);
        }
    } else if (alloc_type == ALLOC_HOST_PTR) {
        host_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);
        if (err < 0) {
            printf("Error clCreateBuffer host %d\n", err);
            exit(1);
        }
        *ptr = clEnqueueMapBuffer(queue, host_mem, CL_TRUE, flags, 0, size, 0, NULL, NULL, &err);
        if (err < 0) {
            printf("Error clEnqueueMapBuffer host %d\n", err);
            exit(1);
        }
    }
    return host_mem;
}

/** allocates aligned memory on host only if required, returns NULL otherwise */
cl_mem alloc_dev(cl_context ctx, cl_mem_flags rw, size_t size, void *host_ptr, int alloc_type)
{
    cl_mem dev_mem;
    cl_int err;
    cl_mem_flags flags = rw;
    void *ptr = host_ptr;

    if (alloc_type == NATIVE) {
        ptr = NULL;
    } else if (alloc_type == NATIVE_COPY) {
        flags |= CL_MEM_COPY_HOST_PTR;
    } else if (alloc_type == ALLOC_HOST_PTR) {
        //flags |= CL_MEM_ALLOC_HOST_PTR;
        //flags |= CL_MEM_USE_HOST_PTR;
        ptr = NULL;
    } else {
        printf("wrong alloc_type %d\n", alloc_type);
        exit(1);
    }

    dev_mem = clCreateBuffer(ctx, flags, size, ptr, &err);
    if (err < 0) {
        printf("Error clCreateBuffer dev %d\n", err);
        exit(1);
    }

    return dev_mem;
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        printf("Usage : ./bench $nbElts $batch $platform_id $device_id $alloc_type\n");
        printf(
            "alloc_type is : \n"
            "0 : host aligned and basic opencl memory\n"
            "1 : host aligned and CL_MEM_COPY_HOST_PTR\n"
            "2 : CL_MEM_ALLOC_HOST_PTR|CL_MEM_USE_HOST_PTR\n");
        exit(1);
    }

    int nbElts = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int alloc_type = atoi(argv[5]);
    size_t float_elt_size =
        rounded_size_aligned(nbElts * sizeof(float), CACHE_LINE_SIZE);
    size_t int_elt_size =
        rounded_size_aligned(nbElts * sizeof(int32_t), CACHE_LINE_SIZE);

    float *host_a_f = NULL;
    float *host_b_f = NULL;
    float *host_c_f = NULL;
    float *host_c_ref_f = NULL;
    int32_t *host_a_i = NULL;
    int32_t *host_b_i = NULL;
    int32_t *host_c_i = NULL;
    int32_t *host_c_ref_i = NULL;

    struct timespec start, stop;
    double elapsed = 0.0;

    cl_int err;
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_device_id device;
    int platform_choice = 0, device_choice = 0;

    cl_mem dev_a_f, dev_b_f, dev_c_f, dev_a_i, dev_b_i, dev_c_i;
    cl_event event[MAX_EVENTS];
    cl_ulong t_begin_htod, t_end_htod, t_begin_kernel, t_end_kernel, t_begin_dtoh,
        t_end_dtoh;

    cl_program program;
    cl_kernel kernel;
    size_t local_size, global_size;

    platform_choice = atoi(argv[3]);
    device_choice = atoi(argv[4]);
    device = create_device(platform_choice, device_choice);

    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        printf("Error clCreateContext %d, line %d\n", err, __LINE__);
        exit(1);
    }

    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err < 0) {
        printf("Error clCreateCommandQueue %d, line %d\n", err, __LINE__);
        exit(1);
    }

    /** cl_mem for opencl runtime allocated buffer for pinned memory*/
    cl_mem h_a_f, h_b_f, h_c_f, h_a_i, h_b_i, h_c_i;
    h_a_f = alloc_host(ctx, queue, CL_MAP_WRITE, (void **) &host_a_f, float_elt_size, alloc_type);
    h_b_f = alloc_host(ctx, queue, CL_MAP_WRITE, (void **) &host_b_f, float_elt_size, alloc_type);
    h_c_f = alloc_host(ctx, queue, CL_MAP_READ, (void **) &host_c_f, float_elt_size, alloc_type);
    alloc_host(ctx, queue, CL_MAP_READ, (void **) &host_c_ref_f, float_elt_size, NATIVE);
    h_a_i = alloc_host(ctx, queue, CL_MAP_WRITE, (void **) &host_a_i, float_elt_size, alloc_type);
    h_b_i = alloc_host(ctx, queue, CL_MAP_WRITE, (void **) &host_b_i, float_elt_size, alloc_type);
    h_c_i = alloc_host(ctx, queue, CL_MAP_READ, (void **) &host_c_i, float_elt_size, alloc_type);
    alloc_host(ctx, queue, CL_MAP_READ, (void **) &host_c_ref_i, float_elt_size, NATIVE);

    printf("Host alloc done\n");

    // CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR behave differently on different targets
    dev_a_f = alloc_dev(ctx, CL_MEM_READ_ONLY, float_elt_size, &host_a_f, alloc_type);
    dev_b_f = alloc_dev(ctx, CL_MEM_READ_ONLY, float_elt_size, &host_b_f, alloc_type);
    dev_c_f = alloc_dev(ctx, CL_MEM_WRITE_ONLY, float_elt_size, &host_c_f, alloc_type);
    dev_a_i = alloc_dev(ctx, CL_MEM_READ_ONLY, float_elt_size, &host_a_i, alloc_type);
    dev_b_i = alloc_dev(ctx, CL_MEM_READ_ONLY, float_elt_size, &host_b_i, alloc_type);
    dev_c_i = alloc_dev(ctx, CL_MEM_WRITE_ONLY, float_elt_size, &host_c_i, alloc_type);

    printf("GPU alloc done\n");

    program =
        build_program(ctx, device, "./simd_utils_kernel.cl", "-cl-mad-enable -cl-std=CL1.2 -cl-no-signed-zeros");

    write_binaries(program, 1, platform_choice);
    /* Create a kernel */
    kernel = clCreateKernel(program, "kernel_test", &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        exit(1);
    };

    /* Create kernel arguments */
    int type = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_a_f);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_b_f);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_c_f);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &nbElts);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &batch);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &type);
    if (err < 0) {
        printf("Error clSetKernelArg %d, line %d\n", err, __LINE__);
        exit(1);
    }

    global_size = nbElts;

    for (int i = 0; i < nbElts; i++) {
        host_a_f[i] = (float) (i + 1.0f) / 100000.0f;
        host_b_f[i] = 1.3f * ((float) i) / 10.0f;
        host_c_ref_f[i] = host_a_f[i] + host_b_f[i];
    }

    // Bake
    ////////////////////
    err = clEnqueueWriteBuffer(queue, dev_a_f, CL_FALSE, 0, float_elt_size,
                               host_a_f, 0, NULL, &event[0]);
    err |= clEnqueueWriteBuffer(queue, dev_b_f, CL_FALSE, 0, float_elt_size,
                                host_b_f, 0, NULL, &event[1]);
    if (err < 0) {
        printf("Error clEnqueueWriteBuffer %d, line %d\n", err, __LINE__);
        exit(1);
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                 NULL, &event[2]);
    if (err < 0) {
        printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
        exit(1);
    }

    err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                              host_c_f, 0, NULL, &event[3]);
    if (err < 0) {
        printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
        exit(1);
    }

    clWaitForEvents(4, event);
    // CL_PROFILING_COMMAND_QUEUED,  CL_PROFILING_COMMAND_SUBMIT ,
    // CL_PROFILING_COMMAND_START,  CL_PROFILING_COMMAND_END
    err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &t_begin_htod, NULL);
    err |= clGetEventProfilingInfo(event[1], CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &t_end_htod, NULL);
    err |= clGetEventProfilingInfo(event[2], CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &t_begin_kernel, NULL);
    err |= clGetEventProfilingInfo(event[2], CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &t_end_kernel, NULL);
    err |= clGetEventProfilingInfo(event[3], CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &t_begin_dtoh, NULL);
    err |= clGetEventProfilingInfo(event[3], CL_PROFILING_COMMAND_END,
                                   sizeof(cl_ulong), &t_end_dtoh, NULL);
    if (err < 0) {
        printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
        exit(1);
    }

    printf("Elapsed : htod %0.3lf kernel %0.3lf dtoh %0.3lf\n", (t_end_htod - t_begin_htod) * 1e-3,
           (t_end_kernel - t_begin_kernel) * 1e-3, (t_end_dtoh - t_begin_dtoh) * 1e-3);
    ///////////////////


    printf("\n");
    /// Sincos /////
    type = 0;

    err = clSetKernelArg(kernel, 5, sizeof(int), &type);

    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("sincos_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        sincos128f(host_a_f, host_b_f, host_c_ref_f, nbElts);
        //exp_128f(host_a_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("sincos128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 1;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);

    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("sincos_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    printf("\n");
    /// LN /////
    type = 2;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("ln_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        ln_128f(host_a_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("ln_128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 3;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("ln_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }
    l2_err(host_c_ref_f, host_c_f, nbElts);

    printf("\n");
    /// EXP /////

    type = 4;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("expf_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        exp_128f(host_a_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("exp_128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 5;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("expf_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    printf("\n");
    /// TAN /////

    type = 6;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("tanf_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        tan128f(host_a_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("tan128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);


    type = 7;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("tanf_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    // TAN GPU VEC2 //
    type = 14;
    global_size = nbElts / 2;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("tanf_vec2_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 16;
    global_size = nbElts / 2;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("tanf_vec2_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    // TAN GPU VEC4 //
    type = 15;
    global_size = nbElts / 4;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("tanf_vec4_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 17;
    global_size = nbElts / 4;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("tanf_vec4_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    /*for(int i = 0; i < nbElts; i++){
        printf("%d : %f %f %f\n", i, host_a_f[i], host_c_f[i], host_c_ref_f[i]);
    }*/


    global_size = nbElts;
    printf("\n");
    /// ATAN /////

    type = 8;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("atanf_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        atan128f(host_a_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("atan128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);


    type = 9;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("atanf_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);


    printf("\n");
    /// ATAN2 /////

    type = 10;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("atan2f_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        atan2128f(host_a_f, host_b_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("atan2128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 11;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("atan2f_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);


    printf("\n");
    /// ASIN /////

    type = 12;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("asinf_opencl : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);

        clock_gettime(CLOCK_REALTIME, &start);
        asin128f(host_a_f, host_c_ref_f, nbElts);
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsed = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) * 1e-3;
        printf("asin128f %d %0.3lf\n", nbElts, elapsed);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    type = 13;
    err = clSetKernelArg(kernel, 5, sizeof(int), &type);
    for (int loop = 0; loop < 5; loop++) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                                     NULL, &event[0]);
        if (err < 0) {
            printf("Error clEnqueueNDRangeKernel %d, line %d\n", err, __LINE__);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, dev_c_f, CL_TRUE, 0, float_elt_size,
                                  host_c_f, 0, NULL, &event[1]);
        if (err < 0) {
            printf("Error clEnqueueReadBuffer %d, line %d\n", err, __LINE__);
            exit(1);
        }
        clWaitForEvents(2, event);

        err = clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &t_begin_kernel, NULL);
        err |= clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_END,
                                       sizeof(cl_ulong), &t_end_kernel, NULL);
        if (err < 0) {
            printf("Error clGetEventProfilingInfo %d, line %d\n", err, __LINE__);
            exit(1);
        }

        printf("asinf_opencl_native : %d %0.3lf\n", nbElts, (t_end_kernel - t_begin_kernel) * 1e-3);
    }

    l2_err(host_c_ref_f, host_c_f, nbElts);

    free(host_a_f);
    free(host_b_f);
    free(host_c_f);
    free(host_c_ref_f);
    free(host_a_i);
    free(host_b_i);
    free(host_c_i);
    free(host_c_ref_i);

    clReleaseMemObject(dev_a_f);
    clReleaseMemObject(dev_b_f);
    clReleaseMemObject(dev_c_f);

    clReleaseMemObject(dev_a_i);
    clReleaseMemObject(dev_b_i);
    clReleaseMemObject(dev_c_i);

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
