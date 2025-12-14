https://docs.google.com/document/d/1M1K7e7DKsQZQ6I2NXVN2B3HfL8vPFE_jBp16hSNpAdk/edit?tab=t.0

Build Commands:

Example with Kernel 3: 

source rock/bin/activate # Only done once when you log into server
hipcc --offload-arch=gfx1151 -D__AMDGCN_WAVEFRONT_SIZE=32 -O3 sgemm_shared_memory.cpp -o sgemm_shared_memory # compile command
./sgemm_shared_memory 4096 4096 4096 # execution command

