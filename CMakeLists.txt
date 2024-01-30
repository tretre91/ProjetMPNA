cmake_minimum_required(VERSION 3.14)

project(PRR LANGUAGES C)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(deps/argtable)

set(SOURCES src/main.c src/eig.c)

add_library(common INTERFACE)
target_link_libraries(common INTERFACE argtable3 BLAS::BLAS LAPACK::LAPACK)
target_include_directories(common INTERFACE include/)
target_compile_features(common INTERFACE c_std_11)

if (NOT MSVC)
  target_link_libraries(common INTERFACE m)
  target_compile_options(common INTERFACE -Wall -Wextra)
endif()

if (USE_ASAN)
  target_compile_options(common INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_libraries(common INTERFACE asan ubsan)
  target_link_options(common INTERFACE -fsanitize=address,undefined)
endif()

if (USE_MAQAO)
  target_compile_options(common INTERFACE -g -fno-omit-frame-pointer -march=native -O3)
endif()

add_executable(prr src/main.c src/eig.c)
target_link_libraries(prr PRIVATE common OpenMP::OpenMP_C)

add_executable(prr_blas src/main.c src/eig.c)
target_link_libraries(prr_blas PRIVATE common OpenMP::OpenMP_C)
target_compile_definitions(prr_blas PRIVATE -DUSE_CBLAS)

add_executable(prr_seq src/prr_seq.c src/eig.c)
target_link_libraries(prr_seq PRIVATE common)
