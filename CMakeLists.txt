cmake_minimum_required(VERSION 3.14)

project(PRR LANGUAGES C)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(deps/argtable)

set(SOURCES src/main.c src/eig.c)

add_executable(prr ${SOURCES})
target_link_libraries(prr PRIVATE argtable3 BLAS::BLAS LAPACK::LAPACK OpenMP::OpenMP_C)
target_include_directories(prr PRIVATE include/)
target_compile_features(prr PRIVATE c_std_99)

if (NOT MSVC)
  target_link_libraries(prr PRIVATE m)
  target_compile_options(prr PRIVATE -Wall -Wextra)
endif()

if (USE_ASAN)
  target_compile_options(prr PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_libraries(prr PRIVATE asan)
target_link_options(prr PRIVATE -fsanitize=address)
endif()
