# =============================================================================
# Generic Setup
# =============================================================================

# Setting environment rules
cmake_minimum_required (VERSION 3.20)
#Setting CMAKE Toolchain file for vcpkg integration
set(CMAKE_TOOLCHAIN_FILE "C:/Users/jacks/vcpkg/scripts/buildsystems/vcpkg.cmake")
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
project ("sakusaku")

# IMPORTANT! If we don't do this, cef files use the wrong version of max() and don't compile
add_compile_definitions(NOMINMAX) # Prevents Windows min/max macros

# Match CEF's runtime library (static multi-threaded)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Set the standard to C++20
# Set the standard globally to C++20 for all sub-folders/targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Forces it to error if C++20 isn't available
set(CMAKE_CXX_EXTENSIONS OFF)  # Stick to clean, standard C++ (no compiler-specific stuff)    

# =============================================================================
# Building Executable
# =============================================================================

# Un-comment out the WIN32 line to make it run w "src/LLMManager.cpp"ithout console!
add_executable(sakusaku #WIN32
    "main.cpp"
    "sqlite3.h"
    "sqlite3.c" 
    "src/RawData.h"
    "src/DatabaseManager.h"
    "src/DatabaseManager.cpp" 
    "src/LLMManager.h"
    "src/LLMManager.cpp")

# setting up local LLM 
set(LLAMA_DIR "C:/Users/jacks/OneDrive/Documents/CompSci/ohyeahwooyeah/LocalLLMs/llama.cpp")

target_include_directories(sakusaku 
    PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(sakusaku PRIVATE
    ${LLAMA_DIR}/build/src/Release/llama.lib
    ${LLAMA_DIR}/build/ggml/src/Release/ggml.lib
    ${LLAMA_DIR}/build/ggml/src/Release/ggml-base.lib
    ${LLAMA_DIR}/build/ggml/src/Release/ggml-cpu.lib
)

set(LLAMA_BIN "C:/Users/jacks/OneDrive/Documents/CompSci/ohyeahwooyeah/LocalLLMs/llama.cpp/build/bin/Release")

add_custom_command(TARGET sakusaku POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LLAMA_BIN}/llama.dll
        ${LLAMA_BIN}/ggml.dll
        ${LLAMA_BIN}/ggml-base.dll
        ${LLAMA_BIN}/ggml-cpu.dll
        ${LLAMA_BIN}/ggml-cuda.dll
        $<TARGET_FILE_DIR:sakusaku>
)