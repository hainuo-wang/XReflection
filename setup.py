from setuptools import setup, Extension, find_packages
import sys
import os

# --- Conditional compilation logic ---
# Check environment variable to optionally skip C++ extension compilation
SKIP_COMPILATION_ENV_VAR = 'COMPILE'
SKIP_COMPILATION = os.environ.get(SKIP_COMPILATION_ENV_VAR, '0') == '0'

ext_modules_list = []
setup_requires_list = []
# Base install_requires; pybind11 will be added conditionally
install_requires_list = [
    'opencv-python',
    'pillow',
    'scipy',
    'tqdm',
    'torch',
    'torchvision',
    'pytorch-msssim',
    'lightning',
    'tensorboard',
    'wandb>=0.12.10'
]

if SKIP_COMPILATION:
    print(f"INFO: Skipping C++ extension compilation because environment variable {SKIP_COMPILATION_ENV_VAR}=0 is set.")
else:
    # If we are compiling, pybind11 is a build-time dependency.
    # It's also good practice to list it in install_requires if the compiled
    # module might have runtime dependencies on pybind11 stubs, or for consistency.
    setup_requires_list.append('pybind11')
    install_requires_list.insert(0, 'pybind11') # Add pybind11 to install_requires

    try:
        import pybind11 # pybind11.get_include() needs this.
                       # setup_requires should ensure pybind11 is available.
    except ImportError:
        print("ERROR: pybind11 is required to compile C++ extensions but is not found.")
        print(f"Please install pybind11 (e.g., 'pip install pybind11') or set {SKIP_COMPILATION_ENV_VAR}=1 to skip compilation.")
        # If pybind11 is critical and missing, you might want to exit or raise an error.
        # For now, if it fails, ext_modules_list will remain empty or cause error later.
        # To be robust, we can clear lists if import fails, effectively skipping compilation.
        ext_modules_list = []
        if 'pybind11' in setup_requires_list:
            setup_requires_list.remove('pybind11')
        if 'pybind11' in install_requires_list:
            install_requires_list.remove('pybind11')
        print("INFO: Proceeding without C++ extensions due to missing pybind11.")

    if 'pybind11' in setup_requires_list: # Proceed only if pybind11 setup is intended
        # Path to the C++ source file
        cpp_source = os.path.join('xreflection', 'ops', 'data_synthesis', 'reflection_module.cpp')

        # Check if the C++ source file exists
        if not os.path.exists(cpp_source):
            print(f"WARNING: C++ source file not found at {cpp_source}. Skipping C++ extension compilation.")
            ext_modules_list = []
            if 'pybind11' in setup_requires_list:
                setup_requires_list.remove('pybind11')
            if 'pybind11' in install_requires_list:
                install_requires_list.remove('pybind11')
        else:
            ext_modules_list = [
                Extension(
                    'xreflection.ops.data_synthesis.reflection_module',
                    [cpp_source],
                    include_dirs=[
                        pybind11.get_include(),
                        '/usr/include/opencv4',  # Note: This path is system-specific
                    ],
                    libraries=[
                        'opencv_core',
                        'opencv_imgproc',
                    ],
                    library_dirs=[
                        '/usr/lib/x86_64-linux-gnu',  # Note: This path is system-specific
                    ],
                    language='c++',
                    extra_compile_args=[
                        '-Ofast', '-std=c++14', '-fPIC', '-DPYBIND11_DETAILED_ERROR_MESSAGES'
                    ],
                ),
            ]
# --- End of conditional compilation logic ---

setup(
    name='xreflection',
    version='0.1',
    description='An Easy-to-use Toolbox for Single-image Reflection Removal',
    author='Mingjia Li, Hainuo Wang, Jiarui Wang, Qiming Hu and Xiaojie Guo',
    packages=find_packages(),
    ext_modules=ext_modules_list,
    install_requires=install_requires_list,
    setup_requires=setup_requires_list,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
