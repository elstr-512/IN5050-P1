# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build

# Include any dependencies generated for this target.
include CMakeFiles/c63pred.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/c63pred.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/c63pred.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c63pred.dir/flags.make

CMakeFiles/c63pred.dir/codegen:
.PHONY : CMakeFiles/c63pred.dir/codegen

CMakeFiles/c63pred.dir/c63dec.cu.o: CMakeFiles/c63pred.dir/flags.make
CMakeFiles/c63pred.dir/c63dec.cu.o: /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/c63dec.cu
CMakeFiles/c63pred.dir/c63dec.cu.o: CMakeFiles/c63pred.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/c63pred.dir/c63dec.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/c63pred.dir/c63dec.cu.o -MF CMakeFiles/c63pred.dir/c63dec.cu.o.d -x cu -c /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/c63dec.cu -o CMakeFiles/c63pred.dir/c63dec.cu.o

CMakeFiles/c63pred.dir/c63dec.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/c63pred.dir/c63dec.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/c63pred.dir/c63dec.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/c63pred.dir/c63dec.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target c63pred
c63pred_OBJECTS = \
"CMakeFiles/c63pred.dir/c63dec.cu.o"

# External object files for target c63pred
c63pred_EXTERNAL_OBJECTS =

c63pred: CMakeFiles/c63pred.dir/c63dec.cu.o
c63pred: CMakeFiles/c63pred.dir/build.make
c63pred: CMakeFiles/c63pred.dir/compiler_depend.ts
c63pred: libc63.a
c63pred: CMakeFiles/c63pred.dir/linkLibs.rsp
c63pred: CMakeFiles/c63pred.dir/objects1.rsp
c63pred: CMakeFiles/c63pred.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable c63pred"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c63pred.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/c63pred.dir/build: c63pred
.PHONY : CMakeFiles/c63pred.dir/build

CMakeFiles/c63pred.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c63pred.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c63pred.dir/clean

CMakeFiles/c63pred.dir/depend:
	cd /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles/c63pred.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/c63pred.dir/depend

