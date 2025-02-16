IGR Final Project

architecture:
- `neuralSubDiv/`: contains the code for the neural subdivision algorithm

[explaination video](https://www.youtube.com/watch?v=JUiXMmQ0sIA)

check license original

pip install --target 'C:\\Users\\benjamin\\AppData\\Roaming\\Blender Foundation\\Blender\\4.3\\scripts\\modules'

 C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin> .\python.exe -m pip install torch --target "c:\program files\blender foundation\blender 4.3\4.3\python\lib\site-packages"



cd 09_random_subdiv_remesh
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release