# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yf-wu/openacc_comp_cu_c2t

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yf-wu/openacc_comp_cu_c2t/build

# Include any dependencies generated for this target.
include CMakeFiles/project.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/project.dir/flags.make

CMakeFiles/project.dir/main_acc.cpp.o: CMakeFiles/project.dir/flags.make
CMakeFiles/project.dir/main_acc.cpp.o: ../main_acc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yf-wu/openacc_comp_cu_c2t/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/project.dir/main_acc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/project.dir/main_acc.cpp.o -c /home/yf-wu/openacc_comp_cu_c2t/main_acc.cpp

CMakeFiles/project.dir/main_acc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/project.dir/main_acc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yf-wu/openacc_comp_cu_c2t/main_acc.cpp > CMakeFiles/project.dir/main_acc.cpp.i

CMakeFiles/project.dir/main_acc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/project.dir/main_acc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yf-wu/openacc_comp_cu_c2t/main_acc.cpp -o CMakeFiles/project.dir/main_acc.cpp.s

CMakeFiles/project.dir/main_acc.cpp.o.requires:

.PHONY : CMakeFiles/project.dir/main_acc.cpp.o.requires

CMakeFiles/project.dir/main_acc.cpp.o.provides: CMakeFiles/project.dir/main_acc.cpp.o.requires
	$(MAKE) -f CMakeFiles/project.dir/build.make CMakeFiles/project.dir/main_acc.cpp.o.provides.build
.PHONY : CMakeFiles/project.dir/main_acc.cpp.o.provides

CMakeFiles/project.dir/main_acc.cpp.o.provides.build: CMakeFiles/project.dir/main_acc.cpp.o


# Object files for target project
project_OBJECTS = \
"CMakeFiles/project.dir/main_acc.cpp.o"

# External object files for target project
project_EXTERNAL_OBJECTS =

project: CMakeFiles/project.dir/main_acc.cpp.o
project: CMakeFiles/project.dir/build.make
project: cuda/libgpu.so
project: /usr/local/cuda/lib64/libcudart_static.a
project: /usr/lib/x86_64-linux-gnu/librt.so
project: CMakeFiles/project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yf-wu/openacc_comp_cu_c2t/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/project.dir/build: project

.PHONY : CMakeFiles/project.dir/build

CMakeFiles/project.dir/requires: CMakeFiles/project.dir/main_acc.cpp.o.requires

.PHONY : CMakeFiles/project.dir/requires

CMakeFiles/project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/project.dir/clean

CMakeFiles/project.dir/depend:
	cd /home/yf-wu/openacc_comp_cu_c2t/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yf-wu/openacc_comp_cu_c2t /home/yf-wu/openacc_comp_cu_c2t /home/yf-wu/openacc_comp_cu_c2t/build /home/yf-wu/openacc_comp_cu_c2t/build /home/yf-wu/openacc_comp_cu_c2t/build/CMakeFiles/project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/project.dir/depend

