cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(NVRTC_Parallelism LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDAToolkit)

option(ENABLE_PIX_RUNTIME "Build with WinPixEventRuntime" ON)

if(ENABLE_PIX_RUNTIME)
    set(WinPixEventRuntime_DIR external/pix)
    find_package(WinPixEventRuntime REQUIRED)
endif()

set(target_name "NVRTC_Parallelism")
add_executable(${target_name} main.cpp)

set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NVRTC_Parallelism)
    set_property(TARGET ${target_name} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
endif()

set_target_properties(${target_name} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)

if(ENABLE_PIX_RUNTIME)
	target_compile_options(${target_name} PUBLIC -DUSE_PIX=1 -DENABLE_PIX_RUNTIME=1)
	target_link_libraries(${target_name} PUBLIC PIX::WinPixEventRuntime)
	add_custom_command(TARGET ${target_name} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${WinPixEventRuntime_DLL}
		$<TARGET_FILE_DIR:${target_name}>
	)
endif()