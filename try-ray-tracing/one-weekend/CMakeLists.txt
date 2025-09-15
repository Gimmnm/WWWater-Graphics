cmake_minimum_required(VERSION 3.15)

# 子项目：Ray Tracing in One Weekend（章节代码在此目录演进）
project(rtweekend_one_weekend LANGUAGES CXX)

# 确保使用与顶层一致的标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 当前可执行文件（后续新增 .cpp 会被一并加入）
file(GLOB ONE_WEEKEND_SOURCES CONFIGURE_DEPENDS
  "*.cpp" "*.cc" "*.cxx"
)

add_executable(rtweekend ${ONE_WEEKEND_SOURCES})

# 让编译器能找到当前目录下的头文件（支持 #include <vec3.h> 等）
target_include_directories(rtweekend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 尽量保证在不同平台的默认可运行设置
if(APPLE)
  # 在 macOS 下优先生成可在命令行运行的控制台程序
  set_target_properties(rtweekend PROPERTIES MACOSX_BUNDLE FALSE)
endif()

# 可选：为 Debug 开启更多信息
target_compile_definitions(rtweekend PRIVATE $<$<CONFIG:Debug>:RTW_DEBUG=1>)

# Convenience targets to generate and open the PPM image (use absolute paths)
set(RTW_IMG ${CMAKE_CURRENT_BINARY_DIR}/image.ppm)

if(UNIX)
  # Generate image.ppm via an OUTPUT rule so that dependencies and
  # working directories are handled robustly across IDEs.
  add_custom_command(
    OUTPUT ${RTW_IMG}
    COMMAND ${CMAKE_COMMAND} -E echo "Rendering ${RTW_IMG}"
    COMMAND /bin/sh -c "$<TARGET_FILE:rtweekend> > ${RTW_IMG}"
    DEPENDS rtweekend
    VERBATIM
  )

  # 可选：将 image 设为默认构建目标
  option(RTW_IMAGE_AS_DEFAULT "Build image.ppm when building ALL" OFF)
  if(RTW_IMAGE_AS_DEFAULT)
    add_custom_target(image ALL DEPENDS ${RTW_IMG})
  else()
    add_custom_target(image DEPENDS ${RTW_IMG})
  endif()

  if(APPLE)
    add_custom_target(open_image
      COMMAND open ${RTW_IMG}
      DEPENDS image
      COMMENT "Opening ${RTW_IMG}"
    )
  else()
    add_custom_target(open_image
      COMMAND xdg-open ${RTW_IMG}
      DEPENDS image
      COMMENT "Opening ${RTW_IMG}"
    )
  endif()
endif()
