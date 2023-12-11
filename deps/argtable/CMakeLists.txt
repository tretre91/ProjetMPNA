add_library(argtable3 INTERFACE)
target_sources(argtable3 INTERFACE argtable3.c)
target_include_directories(argtable3 INTERFACE ${CMAKE_CURRENT_LIST_DIR})

