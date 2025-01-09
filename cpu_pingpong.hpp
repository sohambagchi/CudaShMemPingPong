#ifndef CPU_PINGPONG_HPP
#define CPU_PINGPONG_HPP

#include "gpu_pingpong.cuh"

// change ping to pong
void host_ping_function_relaxed_base(std::atomic<uint8_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint8_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PONG, std::memory_order_relaxed, std::memory_order_relaxed)) {
            expected = PING;
        }
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}


void host_ping_function_acqrel_base(std::atomic<uint8_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint8_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PONG, std::memory_order_acq_rel, std::memory_order_acquire)) {
            expected = PING;
        }
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

void host_ping_function_relaxed_decoupled(std::atomic<uint8_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint8_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_relaxed) != expected) {
            expected = PING;
        }
        flag->store(PONG, std::memory_order_relaxed);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

void host_ping_function_acqrel_decoupled(std::atomic<uint8_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint8_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_acquire) != expected) {
            expected = PING;
        }
        flag->store(PONG, std::memory_order_release);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}


void host_pong_function_relaxed_base(std::atomic<uint8_t> *flag) {
    uint8_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PING, std::memory_order_relaxed, std::memory_order_relaxed)) {
            expected = PONG;
        }
    }
}

void host_pong_function_relaxed_decoupled(std::atomic<uint8_t> *flag) {
    uint8_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_relaxed) != expected) {
            expected = PONG;
        }
        // std::cout << i * 1000000. << std::endl;
        flag->store(PING, std::memory_order_relaxed);
    }
}


void host_pong_function_acqrel_base(std::atomic<uint8_t> *flag) {
    uint8_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PING, std::memory_order_acq_rel, std::memory_order_acquire)) {
            expected = PONG;
        }
    }
}

void host_pong_function_acqrel_decoupled(std::atomic<uint8_t> *flag) {
    uint8_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_acquire) != expected) {
            expected = PONG;
        }
        // std::cout << i * 1000000. << std::endl;
        flag->store(PING, std::memory_order_release);
    }
}

void host_ping_device_pong_assymetric() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    cpu_set_t cpuset;

    uint64_t cpu_time_system;
    std::thread t_system(host_ping_function_relaxed_base, (std::atomic<uint8_t> *) flag_system_relaxed, &cpu_time_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed);
    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device;
    std::thread t_device(host_ping_function_relaxed_base, (std::atomic<uint8_t> *) flag_device_relaxed, &cpu_time_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed);
    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread;
    // std::thread t_thread(host_ping_function_relaxed_base, (std::atomic<uint8_t> *) flag_thread_relaxed, &cpu_time_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed);
    // t_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Relaxed, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    uint64_t cpu_time_system_acqrel;

    std::thread t_system_acqrel(host_ping_function_acqrel_base, (std::atomic<uint8_t> *) flag_system_acqrel, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel);
    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device_acqrel;
    std::thread t_device_acqrel(host_ping_function_acqrel_base, (std::atomic<uint8_t> *) flag_device_acqrel, &cpu_time_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel);
    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread_acqrel;
    // std::thread t_thread_acqrel(host_ping_function_acqrel_base, (std::atomic<uint8_t> *) flag_thread_acqrel, &cpu_time_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel);
    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Acq-Rel, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_relaxed_thread;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_relaxed_device;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_relaxed_system;

    // cudaMallocHost(&flag_relaxed_thread, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_relaxed_device, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_relaxed_system, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_relaxed_thread = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_relaxed_device = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_relaxed_system = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    uint64_t cpu_time_relaxed_system;

    std::thread t_system_relaxed(host_ping_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_relaxed_system, &cpu_time_relaxed_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_relaxed_system);
    t_system_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_relaxed_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_relaxed_device;
    std::thread t_device_relaxed(host_ping_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_relaxed_device, &cpu_time_relaxed_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_relaxed_device);
    t_device_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_relaxed_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_relaxed_thread;
    // std::thread t_thread_relaxed(host_ping_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_relaxed_thread, &cpu_time_relaxed_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_relaxed_base<<<1,1>>>(flag_relaxed_thread);
    // t_thread_relaxed.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Relaxed, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_relaxed_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_acqrel_thread;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_acqrel_device;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_acqrel_system;

    // cudaMallocHost(&flag_acqrel_thread, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_acqrel_device, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_acqrel_system, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_acqrel_thread = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_acqrel_device = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_acqrel_system = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    uint64_t cpu_time_acqrel_system;

    std::thread t_acqrel_system(host_ping_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_acqrel_system, &cpu_time_acqrel_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_acqrel_system);
    t_acqrel_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_acqrel_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_acqrel_device;
    std::thread t_acqrel_device(host_ping_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_acqrel_device, &cpu_time_acqrel_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_acqrel_device);
    t_acqrel_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_acqrel_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_acqrel_thread;

    // std::thread t_acqrel_thread(host_ping_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_acqrel_thread, &cpu_time_acqrel_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_acqrel_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_acqrel_base<<<1,1>>>(flag_acqrel_thread);
    // t_acqrel_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Acq-Rel, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_acqrel_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
}

void device_ping_host_pong_assymetric() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    cpu_set_t cpuset;

    std::thread t_system(host_pong_function_relaxed_base, (std::atomic<uint8_t> *) flag_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed, &time_system_relaxed);

    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Relaxed, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device(host_pong_function_relaxed_base, (std::atomic<uint8_t> *) flag_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed, &time_device_relaxed);

    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Relaxed, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread(host_pong_function_relaxed_base, (std::atomic<uint8_t> *) flag_thread_relaxed);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed, &time_thread_relaxed);

    // t_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Relaxed, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    std::thread t_system_acqrel(host_pong_function_acqrel_base, (std::atomic<uint8_t> *) flag_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel, &time_system_acqrel);

    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_acqrel(host_pong_function_acqrel_base, (std::atomic<uint8_t> *) flag_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel, &time_device_acqrel);

    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_acqrel(host_pong_function_acqrel_base, (std::atomic<uint8_t> *) flag_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel, &time_thread_acqrel);

    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Acq-Rel, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_relaxed_thread;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_relaxed_device;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_relaxed_system;

    // cudaMallocHost(&flag_relaxed_thread, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_relaxed_device, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_relaxed_system, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_relaxed_thread = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_relaxed_device = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_relaxed_system = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    std::thread t_system_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_relaxed_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_relaxed_system;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_relaxed_system, &time_relaxed_system);

    t_system_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Relaxed, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_relaxed_system / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_relaxed_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_relaxed_device;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_relaxed_device, &time_relaxed_device);

    t_device_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Relaxed, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_relaxed_device / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_relaxed_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_relaxed_thread;
    // device_ping_kernel_relaxed_base<<<1,1>>>(flag_relaxed_thread, &time_relaxed_thread);
    
    // t_thread_relaxed.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Relaxed, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_relaxed_thread / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_acqrel_thread;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_acqrel_device;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_acqrel_system;

    // cudaMallocHost(&flag_acqrel_thread, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_acqrel_device, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_acqrel_system, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_acqrel_thread = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_acqrel_device = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_acqrel_system = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    std::thread t_acqrel_system(host_pong_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_acqrel_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_system.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_acqrel_system;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_acqrel_system, &time_acqrel_system);

    t_acqrel_system.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_acqrel_system / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_acqrel_device(host_pong_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_acqrel_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_device.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_acqrel_device;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_acqrel_device, &time_acqrel_device);

    t_acqrel_device.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_acqrel_device / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_acqrel_thread(host_pong_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_acqrel_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_acqrel_thread.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_acqrel_thread;
    // device_ping_kernel_acqrel_base<<<1,1>>>(flag_acqrel_thread, &time_acqrel_thread);

    // t_acqrel_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Acq-Rel, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_acqrel_thread / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
}

void host_ping_device_pong_base() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    
    cpu_set_t cpuset;
    
    uint64_t cpu_time_system;
    std::thread t_system(host_ping_function_relaxed_base, (std::atomic<uint8_t> *) flag_system_relaxed, &cpu_time_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_system_relaxed);
    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed) | Host : " << ((double) (cpu_time_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;


    uint64_t cpu_time_device;
    std::thread t_device(host_ping_function_relaxed_base, (std::atomic<uint8_t> *) flag_device_relaxed, &cpu_time_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_device_relaxed);
    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed) | Host : " << ((double) (cpu_time_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;


    uint64_t cpu_time_thread;
    std::thread t_thread(host_ping_function_relaxed_base, (std::atomic<uint8_t> *) flag_thread_relaxed, &cpu_time_thread);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_thread_relaxed);
    t_thread.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Thread, Relaxed) | Host : " << ((double) (cpu_time_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
    

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    uint64_t cpu_time_system_acqrel;
    std::thread t_system_acqrel(host_ping_function_acqrel_base, (std::atomic<uint8_t> *) flag_system_acqrel, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_system_acqrel);
    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel) | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device_acqrel;
    std::thread t_device_acqrel(host_ping_function_acqrel_base, (std::atomic<uint8_t> *) flag_device_acqrel, &cpu_time_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_device_acqrel);
    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel) | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_thread_acqrel;
    std::thread t_thread_acqrel(host_ping_function_acqrel_base, (std::atomic<uint8_t> *) flag_thread_acqrel, &cpu_time_thread_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_thread_acqrel);
    t_thread_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Thread, Acq-Rel) | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
}

void host_ping_device_pong_decoupled() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    cpu_set_t cpuset;
    
    uint64_t cpu_time_system;
    std::thread t_system(host_ping_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_system_relaxed, &cpu_time_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed);
    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed, Decoupled) | Host : " << ((double) (cpu_time_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device;
    std::thread t_device(host_ping_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_device_relaxed, &cpu_time_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed);
    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed, Decoupled) | Host : " << ((double) (cpu_time_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread;
    // std::thread t_thread(host_ping_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_thread_relaxed, &cpu_time_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed);
    // t_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Relaxed, Decoupled) | Host : " << ((double) (cpu_time_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
    

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    uint64_t cpu_time_system_acqrel;
    std::thread t_system_acqrel(host_ping_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_system_acqrel, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel);
    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel, Decoupled) | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device_acqrel; //-
    std::thread t_device_acqrel(host_ping_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_device_acqrel, &cpu_time_device_acqrel); //-
    CPU_ZERO(&cpuset); //-
    CPU_SET(0, &cpuset); //-
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset); //-
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel); 
    t_device_acqrel.join(); //-
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel, Decoupled) | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread_acqrel;
    // std::thread t_thread_acqrel(host_ping_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_thread_acqrel, &cpu_time_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel);
    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Acq-Rel, Decoupled) | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
}

void device_ping_device_pong_decoupled() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1,0,stream_a>>>(flag_system_acqrel, &time_system_acqrel);
    device_pong_kernel_acqrel_decoupled<<<1,1,0,stream_b>>>(flag_system_acqrel);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (System, Acq-Rel, Decoupled) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1,0,stream_a>>>(flag_device_acqrel, &time_device_acqrel);
    device_pong_kernel_acqrel_decoupled<<<1,1,0,stream_b>>>(flag_device_acqrel);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Device, Acq-Rel, Decoupled) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1,0,stream_a>>>(flag_thread_acqrel, &time_thread_acqrel);
    // device_pong_kernel_acqrel_decoupled<<<1,1,0,stream_b>>>(flag_thread_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Thread, Acq-Rel, Decoupled) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1,0,stream_a>>>(flag_system_relaxed, &time_system_relaxed);
    device_pong_kernel_relaxed_decoupled<<<1,1,0,stream_b>>>(flag_system_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (System, Relaxed, Decoupled) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1,0,stream_a>>>(flag_device_relaxed, &time_device_relaxed);
    device_pong_kernel_relaxed_decoupled<<<1,1,0,stream_b>>>(flag_device_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Device, Relaxed, Decoupled) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_decoupled<<<1,1,0,stream_a>>>(flag_thread_relaxed, &time_thread_relaxed);
    // device_pong_kernel_relaxed_decoupled<<<1,1,0,stream_b>>>(flag_thread_relaxed);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Thread, Relaxed, Decoupled) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

}

void device_ping_device_pong_base() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    
    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_base<<<1,1,0,stream_a>>>(flag_system_acqrel, &time_system_acqrel);
    device_pong_kernel_acqrel_base<<<1,1,0,stream_b>>>(flag_system_acqrel);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (System, Acq-Rel) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_base<<<1,1,0,stream_a>>>(flag_device_acqrel, &time_device_acqrel);
    device_pong_kernel_acqrel_base<<<1,1,0,stream_b>>>(flag_device_acqrel);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Device, Acq-Rel) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_thread_acqrel;
    device_ping_kernel_acqrel_base<<<1,1,0,stream_a>>>(flag_thread_acqrel, &time_thread_acqrel);
    device_pong_kernel_acqrel_base<<<1,1,0,stream_b>>>(flag_thread_acqrel);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Thread, Acq-Rel) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_base<<<1,1,0,stream_a>>>(flag_system_relaxed, &time_system_relaxed);
    device_pong_kernel_relaxed_base<<<1,1,0,stream_b>>>(flag_system_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (System, Relaxed) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_base<<<1,1,0,stream_a>>>(flag_device_relaxed, &time_device_relaxed);
    device_pong_kernel_relaxed_base<<<1,1,0,stream_b>>>(flag_device_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Device, Relaxed) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_thread_relaxed;
    device_ping_kernel_relaxed_base<<<1,1,0,stream_a>>>(flag_thread_relaxed, &time_thread_relaxed);
    device_pong_kernel_relaxed_base<<<1,1,0,stream_b>>>(flag_thread_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Thread, Relaxed) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

}

void device_ping_host_pong_base() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    
    cpu_set_t cpuset;

    std::thread t_system_relaxed(host_pong_function_relaxed_base, (std::atomic<uint8_t> *) flag_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_system_relaxed, &time_system_relaxed);

    t_system_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Relaxed) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_relaxed(host_pong_function_relaxed_base, (std::atomic<uint8_t> *) flag_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_device_relaxed, &time_device_relaxed);

    t_device_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Relaxed) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_thread_relaxed(host_pong_function_relaxed_base, (std::atomic<uint8_t> *) flag_thread_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_thread_relaxed;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_thread_relaxed, &time_thread_relaxed);

    t_thread_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Thread, Relaxed) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    std::thread t_system_acqrel(host_pong_function_acqrel_base, (std::atomic<uint8_t> *) flag_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); 
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_system_acqrel, &time_system_acqrel);

    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_acqrel(host_pong_function_acqrel_base, (std::atomic<uint8_t> *) flag_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_device_acqrel, &time_device_acqrel);

    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_thread_acqrel(host_pong_function_acqrel_base, (std::atomic<uint8_t> *) flag_thread_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_thread_acqrel;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_thread_acqrel, &time_thread_acqrel);

    t_thread_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Thread, Acq-Rel) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
}

void device_ping_host_pong_decoupled() {
    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_relaxed;

    // cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_relaxed = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    
    cpu_set_t cpuset;

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed, &time_system_relaxed);
    std::thread t_system_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);


    cudaDeviceSynchronize();
    t_system_relaxed.join();

    std::cout << "Device-PING Host-PONG (System, Relaxed, Decoupled) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed, &time_device_relaxed);
    std::thread t_device_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);


    cudaDeviceSynchronize();
    t_device_relaxed.join();

    std::cout << "Device-PING Host-PONG (Device, Relaxed, Decoupled) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint8_t> *) flag_thread_relaxed);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed, &time_thread_relaxed);

    // t_thread_relaxed.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Relaxed, Decoupled) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    cuda::atomic<uint8_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint8_t, cuda::thread_scope_system> *flag_system_acqrel;

    // cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    // cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    // cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));
    flag_thread_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_thread>));
    flag_device_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_device>));
    flag_system_acqrel = (cuda::atomic<uint8_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint8_t, cuda::thread_scope_system>));

    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel, &time_system_acqrel);
    std::thread t_system_acqrel(host_pong_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); 
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    

    cudaDeviceSynchronize();
    t_system_acqrel.join();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel, Decoupled) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel, &time_device_acqrel);
    std::thread t_device_acqrel(host_pong_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);


    cudaDeviceSynchronize();
    t_device_acqrel.join();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel, Decoupled) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_acqrel(host_pong_function_acqrel_decoupled, (std::atomic<uint8_t> *) flag_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel, &time_thread_acqrel);

    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Acq-Rel, Decoupled) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
}

#endif // CPU_PINGPONG_HPP