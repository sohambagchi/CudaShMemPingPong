#ifndef GHCONSISTENCYTEST_
#define GHCONSISTENCYTEST_

#include <iostream>
#include <cstdint>
#include <thread>

#include "structs.cuh"
// #include "cpu_data_functions.hpp"
// #include "gpu_data_functions.cuh"
#include "cpu_pingpong.hpp"



/**
 * Parameters:
 *  1. Whether flag and data are in the same cacheline or not
 *      a. Cacheline is 64 bytes  => 32 bytes for flag and 32 bytes for data
 *      b. Cacheline is 128 bytes => 64 bytes for flag and 64 bytes for data
 *  2. Different combinations of flag scopes (cta, block, gpu, sys)
 * */


void run_ping_pong_functions() {
    // std::cout << get_cpu_freq() << std::endl;
    // std::cout << get_gpu_freq() << std::endl;
    host_ping_device_pong_base();
    device_ping_host_pong_base();
    host_ping_device_pong_decoupled();
    device_ping_host_pong_decoupled();
    device_ping_device_pong_base();
    device_ping_device_pong_decoupled();
    // host_ping_host_pong_decoupled();

    host_ping_device_pong_assymetric();
    device_ping_host_pong_assymetric();
}

int main(int argc, char** argv) {

    run_ping_pong_functions();

    return 0;
}




#endif 
