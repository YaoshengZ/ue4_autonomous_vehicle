cmake_minimum_required(VERSION 2.6)

add_executable(lane_assistant lane_assistant.cpp)
target_link_libraries(lane_assistant ${TRONIS_CORE_LIB} ${TRONIS_GRABBER_LIB} ${GLFW_LIB} ${EASY3D_LIB})
target_link_libraries(lane_assistant ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})

if (NOT SDK_MODE)
    add_dependencies(lane_assistant tronis_core tronis_grabber glfw easy3d)
endif()
set_property(TARGET lane_assistant
             PROPERTY EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE)