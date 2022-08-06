#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import subprocess
import sys


def main():
    """
    Script to run library with multple .ini files

    For each <file>, ./run.py is invoked putting output in a separate directory (i.e., output_<file?
    """
    parser = argparse.ArgumentParser(description="Performs multiple experiments")
    parser.add_argument("root_directory", help="The root directory containing the ini files of the experiments")
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-j', help="The number of processes to be used", default=1)
    args = parser.parse_args()

    # The absolute path of the current script
    abs_script = os.path.abspath(sys.argv[0])

    # The root directory of the script
    abs_root = os.path.dirname(abs_script)

    for file in os.listdir(args.root_directory):
        if not file.endswith(".ini"):
            continue
        extra_options = ""
        if args.debug:
            extra_options = extra_options + " -d"
        extra_options = extra_options + " -j" + str(args.j)
        command = os.path.join(abs_root, "run.py") + " -t -c " + os.path.join(args.root_directory, file) + " -o output_" + file + extra_options + " 2>&1 | tee log_" + file.replace(".ini", "")
        print("Running " + command)
        subprocess.call(command, shell=True, executable="/bin/bash")


if __name__ == '__main__':
    main()
