PROJECT(PD-Flow)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4) #I have no idea, mine is 3.0.2
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-arch=sm_20")

#Build scene flow CUDA lib
CUDA_ADD_LIBRARY(pdflow_cudalib pdflow_cudalib.h pdflow_cudalib.cu)


SET(DEFAULT_BUILD_EVALUATOR ON)
SET(BUILD_EVALUATOR ${DEFAULT_BUILD_EVALUATOR} CACHE BOOL "Build the scene flow evaluator for a RGB-D frame pair that uses OpenCV")

#Build the scene flow evaluator
IF (BUILD_EVALUATOR)
	FIND_PACKAGE(OpenCV REQUIRED)
	find_package(Boost REQUIRED python)
	find_package(PythonLibs)
	
	include_directories(${OpenCV_INCLUDE_DIRS})
	include_directories(${Boost_INCLUDE_DIR})
	include_directories(${PYTHON_INCLUDE_DIRS})
	
	ADD_LIBRARY(PySceneFlow SHARED
		main_scene_flow_impair.cpp
		scene_flow_impair.cpp
		scene_flow_impair.h
		python.cpp)
	set_target_properties(PySceneFlow PROPERTIES DEBUG_POSTFIX "D")
	set_target_properties(PySceneFlow PROPERTIES SUFFIX ".pyd")
	install(TARGETS PySceneFlow
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
		
	TARGET_LINK_LIBRARIES(PySceneFlow 
			${OpenCV_LIBS}
			${CUDA_LIBRARIES}
			${Boost_LIBRARIES}
			${PYTHON_LIBRARIES}
			pdflow_cudalib)
	
ENDIF (BUILD_EVALUATOR)

SET(DEFAULT_BUILD_RT_VISUALIZATION ON)
SET(BUILD_RT_VISUALIZATION ${DEFAULT_BUILD_RT_VISUALIZATION} CACHE BOOL "Build the scene flow estimator with real-time visualization that requires MRPT and OpenNI2")

#Build real-time scene flow with visualization
IF (BUILD_RT_VISUALIZATION)
	FIND_PACKAGE(MRPT REQUIRED base gui opengl)

	IF (WIN32)
		INCLUDE_DIRECTORIES("C:/Program Files/OpenNI2/Include")
		LINK_DIRECTORIES("C:/Program Files/OpenNI2/Lib")
		SET(OpenNI_lib "C:/Program Files/OpenNI2/Lib/OpenNI2.lib")
	ENDIF (WIN32)
	
	IF (UNIX)
		SET(OpenNI2_libdir "/usr/lib")
		INCLUDE_DIRECTORIES("C:/Program Files/OpenNI2/Include")
		LINK_DIRECTORIES("C:/Program Files/OpenNI2/Lib")
		SET(OpenNI_lib "C:/Program Files/OpenNI2/Lib/OpenNI2.lib")
	ENDIF (UNIX)

	ADD_EXECUTABLE(Scene-Flow-Visualization  
		main_scene_flow_visualization.cpp
		scene_flow_visualization.cpp
		scene_flow_visualization.h
		legend_pdflow.xpm)
		
	TARGET_LINK_LIBRARIES(Scene-Flow-Visualization
			${MRPT_LIBS}
			${CUDA_LIBRARIES}
			${OpenNI_lib}
			pdflow_cudalib)
ENDIF (BUILD_RT_VISUALIZATION)			

			
# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

