# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/ctx/VIO_homework/HW2/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ctx/VIO_homework/HW2/build

# Include any dependencies generated for this target.
include code_utils/CMakeFiles/polynomial.dir/depend.make

# Include the progress variables for this target.
include code_utils/CMakeFiles/polynomial.dir/progress.make

# Include the compile flags for this target's objects.
include code_utils/CMakeFiles/polynomial.dir/flags.make

code_utils/CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.o: code_utils/CMakeFiles/polynomial.dir/flags.make
code_utils/CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.o: /home/ctx/VIO_homework/HW2/src/code_utils/src/math_utils/Polynomial.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ctx/VIO_homework/HW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object code_utils/CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.o"
	cd /home/ctx/VIO_homework/HW2/build/code_utils && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.o -c /home/ctx/VIO_homework/HW2/src/code_utils/src/math_utils/Polynomial.cpp

code_utils/CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.i"
	cd /home/ctx/VIO_homework/HW2/build/code_utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ctx/VIO_homework/HW2/src/code_utils/src/math_utils/Polynomial.cpp > CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.i

code_utils/CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.s"
	cd /home/ctx/VIO_homework/HW2/build/code_utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ctx/VIO_homework/HW2/src/code_utils/src/math_utils/Polynomial.cpp -o CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.s

# Object files for target polynomial
polynomial_OBJECTS = \
"CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.o"

# External object files for target polynomial
polynomial_EXTERNAL_OBJECTS =

/home/ctx/VIO_homework/HW2/devel/lib/libpolynomial.a: code_utils/CMakeFiles/polynomial.dir/src/math_utils/Polynomial.cpp.o
/home/ctx/VIO_homework/HW2/devel/lib/libpolynomial.a: code_utils/CMakeFiles/polynomial.dir/build.make
/home/ctx/VIO_homework/HW2/devel/lib/libpolynomial.a: code_utils/CMakeFiles/polynomial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ctx/VIO_homework/HW2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library /home/ctx/VIO_homework/HW2/devel/lib/libpolynomial.a"
	cd /home/ctx/VIO_homework/HW2/build/code_utils && $(CMAKE_COMMAND) -P CMakeFiles/polynomial.dir/cmake_clean_target.cmake
	cd /home/ctx/VIO_homework/HW2/build/code_utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/polynomial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
code_utils/CMakeFiles/polynomial.dir/build: /home/ctx/VIO_homework/HW2/devel/lib/libpolynomial.a

.PHONY : code_utils/CMakeFiles/polynomial.dir/build

code_utils/CMakeFiles/polynomial.dir/clean:
	cd /home/ctx/VIO_homework/HW2/build/code_utils && $(CMAKE_COMMAND) -P CMakeFiles/polynomial.dir/cmake_clean.cmake
.PHONY : code_utils/CMakeFiles/polynomial.dir/clean

code_utils/CMakeFiles/polynomial.dir/depend:
	cd /home/ctx/VIO_homework/HW2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ctx/VIO_homework/HW2/src /home/ctx/VIO_homework/HW2/src/code_utils /home/ctx/VIO_homework/HW2/build /home/ctx/VIO_homework/HW2/build/code_utils /home/ctx/VIO_homework/HW2/build/code_utils/CMakeFiles/polynomial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : code_utils/CMakeFiles/polynomial.dir/depend

