cmake_minimum_required(VERSION 3.20)

project(vecbase LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Threads REQUIRED)

option(VECBASE_UPDATE_SUBMODULES "Update git submodules during configure" ON)
option(VECBASE_ENABLE_SIMD "Enable SIMD acceleration for vector distance kernels" ON)

if(VECBASE_UPDATE_SUBMODULES)
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE VECBASE_SUBMODULE_RESULT
        )
        if(NOT VECBASE_SUBMODULE_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Failed to initialize git submodules. Run "
                "`git submodule update --init --recursive` and retry."
            )
        endif()
    endif()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dkv/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing dependency at third_party/dkv. Run "
        "`git submodule update --init --recursive`."
    )
endif()

set(DKV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DKV_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(DKV_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(DKV_BUILD_UTILS OFF CACHE BOOL "" FORCE)
set(DKV_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/dkv)

add_library(vecbase
    db/db_impl.cc
    index/distance.cc
    index/hnsw_index.cc
    index/page_manager.cc
    table/buffer_pool.cc
    util/crc32c.cc
    util/lru_k_replacer.cc
)

target_include_directories(vecbase
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(vecbase PUBLIC cxx_std_20)
target_link_libraries(vecbase PRIVATE dkv)

if(VECBASE_ENABLE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|X86_64|amd64|AMD64|i[3-6]86)$")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(vecbase PRIVATE -mavx2 -mfma)
            target_compile_definitions(vecbase PRIVATE VECBASE_SIMD_AVX2_FMA=1)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64|armv7.*|armv8.*)$")
        target_compile_definitions(vecbase PRIVATE VECBASE_SIMD_NEON=1)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
           AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7.*|armv8.*)$")
            target_compile_options(vecbase PRIVATE -mfpu=neon)
        endif()
    endif()
endif()

add_library(vecbase::vecbase ALIAS vecbase)

enable_testing()

add_executable(vecbase_tests
    tests/test_main.cc
)
target_link_libraries(vecbase_tests PRIVATE vecbase dkv)
target_include_directories(vecbase_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dkv/include
)
add_test(NAME vecbase_tests COMMAND vecbase_tests)

add_executable(vecbase_example_basic
    examples/basic_usage.cc
)
target_link_libraries(vecbase_example_basic PRIVATE vecbase dkv)
target_include_directories(vecbase_example_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dkv/include
)

add_executable(vecbase_server
    server/acceptor.cc
    server/commands.cc
    server/main.cc
    server/resp.cc
    server/server.cc
    server/subreactor.cc
)
target_link_libraries(vecbase_server PRIVATE vecbase dkv Threads::Threads)
target_include_directories(vecbase_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dkv/include
)

add_executable(vecbase_cli
    cli/vecbase_cli.cc
)
target_link_libraries(vecbase_cli PRIVATE dkv)
target_include_directories(vecbase_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dkv/include
)
