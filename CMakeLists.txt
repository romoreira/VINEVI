cmake_minimum_required(VERSION 3.17)
project(vinevi C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_COMPILER "/usr/bin/gcc")

add_executable(vinevi watcher.c)