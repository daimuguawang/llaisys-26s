#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <mcr/mc_runtime.h>

// Test: simple vector add kernel
__global__ void vector_add(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        out[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 256;
    float *d_out, *d_a, *d_b;
    mcMalloc((void**)&d_out, n * sizeof(float));
    mcMalloc((void**)&d_a, n * sizeof(float));
    mcMalloc((void**)&d_b, n * sizeof(float));

    // Init host data
    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    mcMemcpy(d_a, h_a, n * sizeof(float), mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, n * sizeof(float), mcMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_out, d_a, d_b, n);

    mcDeviceSynchronize();

    mcMemcpy(h_out, d_out, n * sizeof(float), mcMemcpyDeviceToHost);

    // Verify
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = (float)(i * 3);
        if (fabsf(h_out[i] - expected) > 0.001f) {
            errors++;
            if (errors <= 5) printf("Error at %d: %f vs %f\n", i, h_out[i], expected);
        }
    }

    if (errors == 0) {
        printf("SUCCESS: Vector add kernel works correctly!\n");
    } else {
        printf("FAILED: %d errors found\n", errors);
    }

    mcFree(d_out);
    mcFree(d_a);
    mcFree(d_b);
    free(h_a);
    free(h_b);
    free(h_out);

    return errors;
}
