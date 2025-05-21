OmniViz
=======

Building
--------
```
checkout branch step037-imgui
```
**wasm:**
```
emsdk.ps1 activate
emcmake cmake -B build-web
cmake --build build-web
python -m http.server -d build-web
point browser to http://localhost:8000/App.html
```
**native:**
```
cmake -B build-native
cmake --build build-native
build-native\Debug\App.exe
```