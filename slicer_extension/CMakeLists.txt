cmake_minimum_required(VERSION 3.5)

project(phantom)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/demo")
set(EXTENSION_CATEGORY "phantomAnalysis")  # todo
set(EXTENSION_CONTRIBUTORS "Katie Manduca")
set(EXTENSION_DESCRIPTION "phantom analysis: T1, ADC, and VOI finding")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/demo.png")  # todo
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/demo/Screenshots/1.png")  # todo
set(EXTENSION_DEPENDS "SlicerOpenCV")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

find_package(SlicerOpenCV REQUIRED)

#-----------------------------------------------------------------------------
# Extension modules
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
