cmake_minimum_required(VERSION 3.15)

# 顶层 CMake：管理整个 Try-ray-tracing 工作区
project(TryRayTracing LANGUAGES CXX)

# 选用 C++17（RTIOW 教程示例通常使用 C++11/17，取 17 更宽松）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 常用编译选项（可按需微调）
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 将 one-weekend 子目录加入构建
add_subdirectory(one-weekend)

