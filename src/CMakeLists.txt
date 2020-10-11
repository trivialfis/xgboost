file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/cli_main.cc)

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objxgboost OBJECT)
target_sources(objxgboost PRIVATE ${CPU_SOURCES})
if (USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu *.cuh)
  target_sources(objxgboost PRIVATE ${CUDA_SOURCES})
  target_link_libraries(objxgboost PRIVATE GPUTreeShap)
endif (USE_CUDA)

xgboost_target_set_flags(objxgboost)

find_package(Threads REQUIRED)
target_link_libraries(objxgboost PUBLIC Threads::Threads ${CMAKE_THREAD_LIBS_INIT})

# For MSVC: Call msvc_use_static_runtime() once again to completely
# replace /MD with /MT. See https://github.com/dmlc/xgboost/issues/4462
# for issues caused by mixing of /MD and /MT flags
msvc_use_static_runtime()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")

#-- End object library
