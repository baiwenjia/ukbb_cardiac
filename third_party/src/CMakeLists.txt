cmake_minimum_required(VERSION 2.8)

# The directory for compiled executable files
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# MIRTK
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(MIRTK_SOURCE_DIR /vol/biomedic2/wbai/git/MIRTK)
set(MIRTK_BINARY_DIR /vol/biomedic2/wbai/git/MIRTK_bin)

include_directories(${MIRTK_SOURCE_DIR}/Modules/Common/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Image/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/ImageIO/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Numerics/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/PointSet/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Registration/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Transformation/include)
include_directories(${MIRTK_BINARY_DIR}/include)

link_directories(${MIRTK_BINARY_DIR}/lib)
link_libraries(MIRTKCommon MIRTKNumerics MIRTKImage MIRTKIO MIRTKPointSet MIRTKRegistration MIRTKTransformation)

# VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
link_libraries(${VTK_LIBRARIES})

# Executables
ADD_EXECUTABLE(average_3d_ffd average_3d_ffd.cc)
