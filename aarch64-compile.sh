export cross_compile_toolchain=/data/linaro/aarch64/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/
cd build-aarch64
cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-g++ -DCMAKE_INSTALL_PREFIX=../install-aarch64