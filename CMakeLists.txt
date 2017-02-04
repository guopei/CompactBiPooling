CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

FIND_PACKAGE(Torch REQUIRED)

FIND_PACKAGE(CUDA 6.5 REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--use_fast_math;-O3")
MESSAGE("Compiling for CUDA architecture 3.5")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}")
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/TH")
INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src-cuda cudafft.cu)
SET(luasrc init.lua)
add_definitions( -DTORCH_BUILD )

CUDA_ADD_LIBRARY(cudafft MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(cudafft THC TH cuda cufft)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(cudafft PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

SET(src)
FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} test.lua)
ADD_TORCH_PACKAGE(tcbp "${src}" "${luasrc}"  "Compact Bilinear Pooling")

INSTALL(TARGETS cudafft
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")
