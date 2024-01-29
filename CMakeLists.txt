cmake_minimum_required(VERSION 3.14)

project(PRR LANGUAGES C)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(deps/argtable)

set(SOURCES src/main.c src/eig.c)

add_library(common INTERFACE)
target_sources(common INTERFACE ${SOURCES})
target_link_libraries(common INTERFACE argtable3 BLAS::BLAS LAPACK::LAPACK OpenMP::OpenMP_C)
target_include_directories(common INTERFACE include/)
target_compile_features(common INTERFACE c_std_99)

if (NOT MSVC)
  target_link_libraries(common INTERFACE m)
  target_compile_options(common INTERFACE -Wall -Wextra)
endif()

if (USE_ASAN)
  target_compile_options(common INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_libraries(common INTERFACE asan ubsan)
  target_link_options(common INTERFACE -fsanitize=address,undefined)
endif()

add_executable(prr)
target_link_libraries(prr PRIVATE common)

add_executable(prr_blas)
target_link_libraries(prr_blas PRIVATE common)
target_compile_definitions(prr_blas PRIVATE -DUSE_CBLAS)
