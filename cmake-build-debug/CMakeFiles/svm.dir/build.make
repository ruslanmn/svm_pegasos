# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_COMMAND = /home/kmeansfan/clion-2016.3.4/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/kmeansfan/clion-2016.3.4/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kmeansfan/CLionProjects/svm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kmeansfan/CLionProjects/svm/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/svm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/svm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svm.dir/flags.make

CMakeFiles/svm.dir/main.cpp.o: CMakeFiles/svm.dir/flags.make
CMakeFiles/svm.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/svm.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svm.dir/main.cpp.o -c /home/kmeansfan/CLionProjects/svm/main.cpp

CMakeFiles/svm.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svm.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kmeansfan/CLionProjects/svm/main.cpp > CMakeFiles/svm.dir/main.cpp.i

CMakeFiles/svm.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svm.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kmeansfan/CLionProjects/svm/main.cpp -o CMakeFiles/svm.dir/main.cpp.s

CMakeFiles/svm.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/svm.dir/main.cpp.o.requires

CMakeFiles/svm.dir/main.cpp.o.provides: CMakeFiles/svm.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/svm.dir/build.make CMakeFiles/svm.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/svm.dir/main.cpp.o.provides

CMakeFiles/svm.dir/main.cpp.o.provides.build: CMakeFiles/svm.dir/main.cpp.o


CMakeFiles/svm.dir/mnist_data_loader.cpp.o: CMakeFiles/svm.dir/flags.make
CMakeFiles/svm.dir/mnist_data_loader.cpp.o: ../mnist_data_loader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/svm.dir/mnist_data_loader.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svm.dir/mnist_data_loader.cpp.o -c /home/kmeansfan/CLionProjects/svm/mnist_data_loader.cpp

CMakeFiles/svm.dir/mnist_data_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svm.dir/mnist_data_loader.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kmeansfan/CLionProjects/svm/mnist_data_loader.cpp > CMakeFiles/svm.dir/mnist_data_loader.cpp.i

CMakeFiles/svm.dir/mnist_data_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svm.dir/mnist_data_loader.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kmeansfan/CLionProjects/svm/mnist_data_loader.cpp -o CMakeFiles/svm.dir/mnist_data_loader.cpp.s

CMakeFiles/svm.dir/mnist_data_loader.cpp.o.requires:

.PHONY : CMakeFiles/svm.dir/mnist_data_loader.cpp.o.requires

CMakeFiles/svm.dir/mnist_data_loader.cpp.o.provides: CMakeFiles/svm.dir/mnist_data_loader.cpp.o.requires
	$(MAKE) -f CMakeFiles/svm.dir/build.make CMakeFiles/svm.dir/mnist_data_loader.cpp.o.provides.build
.PHONY : CMakeFiles/svm.dir/mnist_data_loader.cpp.o.provides

CMakeFiles/svm.dir/mnist_data_loader.cpp.o.provides.build: CMakeFiles/svm.dir/mnist_data_loader.cpp.o


CMakeFiles/svm.dir/byteorder_helper.cpp.o: CMakeFiles/svm.dir/flags.make
CMakeFiles/svm.dir/byteorder_helper.cpp.o: ../byteorder_helper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/svm.dir/byteorder_helper.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svm.dir/byteorder_helper.cpp.o -c /home/kmeansfan/CLionProjects/svm/byteorder_helper.cpp

CMakeFiles/svm.dir/byteorder_helper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svm.dir/byteorder_helper.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kmeansfan/CLionProjects/svm/byteorder_helper.cpp > CMakeFiles/svm.dir/byteorder_helper.cpp.i

CMakeFiles/svm.dir/byteorder_helper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svm.dir/byteorder_helper.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kmeansfan/CLionProjects/svm/byteorder_helper.cpp -o CMakeFiles/svm.dir/byteorder_helper.cpp.s

CMakeFiles/svm.dir/byteorder_helper.cpp.o.requires:

.PHONY : CMakeFiles/svm.dir/byteorder_helper.cpp.o.requires

CMakeFiles/svm.dir/byteorder_helper.cpp.o.provides: CMakeFiles/svm.dir/byteorder_helper.cpp.o.requires
	$(MAKE) -f CMakeFiles/svm.dir/build.make CMakeFiles/svm.dir/byteorder_helper.cpp.o.provides.build
.PHONY : CMakeFiles/svm.dir/byteorder_helper.cpp.o.provides

CMakeFiles/svm.dir/byteorder_helper.cpp.o.provides.build: CMakeFiles/svm.dir/byteorder_helper.cpp.o


CMakeFiles/svm.dir/svm.cpp.o: CMakeFiles/svm.dir/flags.make
CMakeFiles/svm.dir/svm.cpp.o: ../svm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/svm.dir/svm.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svm.dir/svm.cpp.o -c /home/kmeansfan/CLionProjects/svm/svm.cpp

CMakeFiles/svm.dir/svm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svm.dir/svm.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kmeansfan/CLionProjects/svm/svm.cpp > CMakeFiles/svm.dir/svm.cpp.i

CMakeFiles/svm.dir/svm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svm.dir/svm.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kmeansfan/CLionProjects/svm/svm.cpp -o CMakeFiles/svm.dir/svm.cpp.s

CMakeFiles/svm.dir/svm.cpp.o.requires:

.PHONY : CMakeFiles/svm.dir/svm.cpp.o.requires

CMakeFiles/svm.dir/svm.cpp.o.provides: CMakeFiles/svm.dir/svm.cpp.o.requires
	$(MAKE) -f CMakeFiles/svm.dir/build.make CMakeFiles/svm.dir/svm.cpp.o.provides.build
.PHONY : CMakeFiles/svm.dir/svm.cpp.o.provides

CMakeFiles/svm.dir/svm.cpp.o.provides.build: CMakeFiles/svm.dir/svm.cpp.o


CMakeFiles/svm.dir/mnist_data_classifier.cpp.o: CMakeFiles/svm.dir/flags.make
CMakeFiles/svm.dir/mnist_data_classifier.cpp.o: ../mnist_data_classifier.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/svm.dir/mnist_data_classifier.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/svm.dir/mnist_data_classifier.cpp.o -c /home/kmeansfan/CLionProjects/svm/mnist_data_classifier.cpp

CMakeFiles/svm.dir/mnist_data_classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svm.dir/mnist_data_classifier.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kmeansfan/CLionProjects/svm/mnist_data_classifier.cpp > CMakeFiles/svm.dir/mnist_data_classifier.cpp.i

CMakeFiles/svm.dir/mnist_data_classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svm.dir/mnist_data_classifier.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kmeansfan/CLionProjects/svm/mnist_data_classifier.cpp -o CMakeFiles/svm.dir/mnist_data_classifier.cpp.s

CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.requires:

.PHONY : CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.requires

CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.provides: CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/svm.dir/build.make CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.provides.build
.PHONY : CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.provides

CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.provides.build: CMakeFiles/svm.dir/mnist_data_classifier.cpp.o


# Object files for target svm
svm_OBJECTS = \
"CMakeFiles/svm.dir/main.cpp.o" \
"CMakeFiles/svm.dir/mnist_data_loader.cpp.o" \
"CMakeFiles/svm.dir/byteorder_helper.cpp.o" \
"CMakeFiles/svm.dir/svm.cpp.o" \
"CMakeFiles/svm.dir/mnist_data_classifier.cpp.o"

# External object files for target svm
svm_EXTERNAL_OBJECTS =

svm: CMakeFiles/svm.dir/main.cpp.o
svm: CMakeFiles/svm.dir/mnist_data_loader.cpp.o
svm: CMakeFiles/svm.dir/byteorder_helper.cpp.o
svm: CMakeFiles/svm.dir/svm.cpp.o
svm: CMakeFiles/svm.dir/mnist_data_classifier.cpp.o
svm: CMakeFiles/svm.dir/build.make
svm: CMakeFiles/svm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable svm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/svm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svm.dir/build: svm

.PHONY : CMakeFiles/svm.dir/build

CMakeFiles/svm.dir/requires: CMakeFiles/svm.dir/main.cpp.o.requires
CMakeFiles/svm.dir/requires: CMakeFiles/svm.dir/mnist_data_loader.cpp.o.requires
CMakeFiles/svm.dir/requires: CMakeFiles/svm.dir/byteorder_helper.cpp.o.requires
CMakeFiles/svm.dir/requires: CMakeFiles/svm.dir/svm.cpp.o.requires
CMakeFiles/svm.dir/requires: CMakeFiles/svm.dir/mnist_data_classifier.cpp.o.requires

.PHONY : CMakeFiles/svm.dir/requires

CMakeFiles/svm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/svm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/svm.dir/clean

CMakeFiles/svm.dir/depend:
	cd /home/kmeansfan/CLionProjects/svm/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kmeansfan/CLionProjects/svm /home/kmeansfan/CLionProjects/svm /home/kmeansfan/CLionProjects/svm/cmake-build-debug /home/kmeansfan/CLionProjects/svm/cmake-build-debug /home/kmeansfan/CLionProjects/svm/cmake-build-debug/CMakeFiles/svm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svm.dir/depend

