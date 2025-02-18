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
include CMakeFiles/c63.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/c63.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/c63.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c63.dir/flags.make

CMakeFiles/c63.dir/codegen:
.PHONY : CMakeFiles/c63.dir/codegen

CMakeFiles/c63.dir/quantdct.cu.o: CMakeFiles/c63.dir/flags.make
CMakeFiles/c63.dir/quantdct.cu.o: /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/quantdct.cu
CMakeFiles/c63.dir/quantdct.cu.o: CMakeFiles/c63.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/c63.dir/quantdct.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/c63.dir/quantdct.cu.o -MF CMakeFiles/c63.dir/quantdct.cu.o.d -x cu -c /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/quantdct.cu -o CMakeFiles/c63.dir/quantdct.cu.o

CMakeFiles/c63.dir/quantdct.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/c63.dir/quantdct.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/c63.dir/quantdct.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/c63.dir/quantdct.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/c63.dir/tables.cu.o: CMakeFiles/c63.dir/flags.make
CMakeFiles/c63.dir/tables.cu.o: /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/tables.cu
CMakeFiles/c63.dir/tables.cu.o: CMakeFiles/c63.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/c63.dir/tables.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/c63.dir/tables.cu.o -MF CMakeFiles/c63.dir/tables.cu.o.d -x cu -c /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/tables.cu -o CMakeFiles/c63.dir/tables.cu.o

CMakeFiles/c63.dir/tables.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/c63.dir/tables.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/c63.dir/tables.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/c63.dir/tables.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/c63.dir/io.cu.o: CMakeFiles/c63.dir/flags.make
CMakeFiles/c63.dir/io.cu.o: /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/io.cu
CMakeFiles/c63.dir/io.cu.o: CMakeFiles/c63.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/c63.dir/io.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/c63.dir/io.cu.o -MF CMakeFiles/c63.dir/io.cu.o.d -x cu -c /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/io.cu -o CMakeFiles/c63.dir/io.cu.o

CMakeFiles/c63.dir/io.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/c63.dir/io.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/c63.dir/io.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/c63.dir/io.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/c63.dir/common.cu.o: CMakeFiles/c63.dir/flags.make
CMakeFiles/c63.dir/common.cu.o: /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/common.cu
CMakeFiles/c63.dir/common.cu.o: CMakeFiles/c63.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/c63.dir/common.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/c63.dir/common.cu.o -MF CMakeFiles/c63.dir/common.cu.o.d -x cu -c /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/common.cu -o CMakeFiles/c63.dir/common.cu.o

CMakeFiles/c63.dir/common.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/c63.dir/common.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/c63.dir/common.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/c63.dir/common.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/c63.dir/me.cu.o: CMakeFiles/c63.dir/flags.make
CMakeFiles/c63.dir/me.cu.o: /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/me.cu
CMakeFiles/c63.dir/me.cu.o: CMakeFiles/c63.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/c63.dir/me.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/c63.dir/me.cu.o -MF CMakeFiles/c63.dir/me.cu.o.d -x cu -c /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/me.cu -o CMakeFiles/c63.dir/me.cu.o

CMakeFiles/c63.dir/me.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/c63.dir/me.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/c63.dir/me.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/c63.dir/me.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target c63
c63_OBJECTS = \
"CMakeFiles/c63.dir/quantdct.cu.o" \
"CMakeFiles/c63.dir/tables.cu.o" \
"CMakeFiles/c63.dir/io.cu.o" \
"CMakeFiles/c63.dir/common.cu.o" \
"CMakeFiles/c63.dir/me.cu.o"

# External object files for target c63
c63_EXTERNAL_OBJECTS =

libc63.a: CMakeFiles/c63.dir/quantdct.cu.o
libc63.a: CMakeFiles/c63.dir/tables.cu.o
libc63.a: CMakeFiles/c63.dir/io.cu.o
libc63.a: CMakeFiles/c63.dir/common.cu.o
libc63.a: CMakeFiles/c63.dir/me.cu.o
libc63.a: CMakeFiles/c63.dir/build.make
libc63.a: CMakeFiles/c63.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CUDA static library libc63.a"
	$(CMAKE_COMMAND) -P CMakeFiles/c63.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c63.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/c63.dir/build: libc63.a
.PHONY : CMakeFiles/c63.dir/build

CMakeFiles/c63.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c63.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c63.dir/clean

CMakeFiles/c63.dir/depend:
	cd /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build /home/littledragon/Skrivebord/IN5050/home_exam_1/IN5050-P1/c63-in-cuda/build/CMakeFiles/c63.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/c63.dir/depend

