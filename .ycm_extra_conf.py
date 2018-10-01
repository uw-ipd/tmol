# This file is NOT licensed under the GPLv3, which is the license for the rest
# of YouCompleteMe.
#
# Here's the license text for this file:
#
# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

from distutils.sysconfig import get_python_inc
import os
import subprocess

DIR_OF_THIS_SCRIPT = os.path.abspath(os.path.dirname(__file__))
DIR_OF_THIRD_PARTY = os.path.join(DIR_OF_THIS_SCRIPT, "tmol/extern")
SOURCE_EXTENSIONS = {"c++": [".cpp", ".cxx", ".cc", "c"], "cuda": [".cu"]}

HEADER_EXTENSIONS = [".h", ".hxx", "hpp", ".hh"]

torch_paths = subprocess.check_output(
    "python -c "
    "'import torch.utils.cpp_extension; "
    'print("\\n".join(torch.utils.cpp_extension.include_paths(True)))\'',
    shell=True,
).splitlines()

tmol_paths = subprocess.check_output(
    "python -c 'import tmol.extern; "
    'print("\\n".join(tmol.include_paths() + tmol.extern.include_paths()))\'',
    shell=True,
).splitlines()


# These are the compilation flags that will be used in case there's no
# compilation database set (by default, one is not set).
# CHANGE THIS LIST OF FLAGS. YES, THIS IS THE DROID YOU HAVE BEEN LOOKING FOR.
flags = ["-fexceptions", "-DNDEBUG", "-std=c++11", "-isystem", get_python_inc()] + [
    "-I%s" % p for p in torch_paths + tmol_paths
]


def is_header_file(filename):
    extension = os.path.splitext(filename)[1]
    return extension in HEADER_EXTENSIONS


def find_corresponding_source_file(filename):
    if is_header_file(filename):
        basename = os.path.splitext(filename)[0]
        for _filetype, extensions in SOURCE_EXTENSIONS.items():
            for extension in extensions:
                replacement_file = basename + extension
                if os.path.exists(replacement_file):
                    return replacement_file
    return filename


def file_language(filename):
    extension = os.path.splitext(filename)[1]

    for language, extensions in SOURCE_EXTENSIONS.items():
        if extension in extensions:
            return language

    return "cpp"


def Settings(**kwargs):
    if kwargs["language"] == "cfamily":
        # If the file is a header, try to find the corresponding source file and
        # retrieve its flags from the compilation database if using one. This is
        # necessary since compilation databases don't have entries for header files.
        # In addition, use this source file as the translation unit. This makes it
        # possible to jump from a declaration in the header file to its definition
        # in the corresponding source file.
        filename = find_corresponding_source_file(kwargs["filename"])

        return {
            "flags": flags + ["-x", file_language(filename)],
            "include_paths_relative_to_dir": DIR_OF_THIS_SCRIPT,
            "override_filename": filename,
        }

    return {}
