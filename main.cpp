/**
 * This file is part of the "Learn WebGPU for C++" book.
 *   https://github.com/eliemichel/LearnWebGPU
 *
 * MIT License
 * Copyright (c) 2022-2025 Elie Michel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <webgpu/webgpu.h>

#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif // __EMSCRIPTEN__

#include <iostream>
#include <cassert>
#include <vector>

/**
 * Utility function to get a WebGPU adapter, so that
 *     WGPUAdapter adapter = requestAdapterSync(options);
 * is roughly equivalent to
 *     const adapter = await navigator.gpu.requestAdapter(options);
 */
WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const * options) {
	// A simple structure holding the local information shared with the
	// onAdapterRequestEnded callback.
	struct UserData {
		WGPUAdapter adapter = nullptr;
		bool requestEnded = false;
	};
	UserData userData;

#ifdef __EMSCRIPTEN__

	// Emscripten still uses an old version of the API

	auto callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata1) {
		UserData& userData = *reinterpret_cast<UserData*>(userdata1);
		if (status == WGPURequestAdapterStatus_Success) {
			userData.adapter = adapter;
		}
		else {
			std::cout << "Could not get WebGPU adapter: " << wgpu::StringView(message) << std::endl;
		}
		userData.requestEnded = true;
	};

	// Call to the WebGPU request adapter procedure
	wgpuInstanceRequestAdapter(instance, options, callback, (void*)&userData);

	// We wait until userData.requestEnded gets true
	while (!userData.requestEnded) {
		emscripten_sleep(100);
	}

#else // __EMSCRIPTEN__

	WGPURequestAdapterCallbackInfo callbackInfo{};
	callbackInfo.nextInChain = nullptr;
	callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
	callbackInfo.userdata1 = (void*)&userData;
	callbackInfo.userdata2 = nullptr;

	// Callback called by wgpuInstanceRequestAdapter when the request returns
	// This is a C++ lambda function, but could be any function defined in the
	// global scope. It must be non-capturing (the brackets [] are empty) so
	// that it behaves like a regular C function pointer, which is what
	// wgpuInstanceRequestAdapter expects (WebGPU being a C API). The workaround
	// is to convey what we want to capture through the pUserData pointer,
	// provided as the last argument of wgpuInstanceRequestAdapter and received
	// by the callback as its last argument.
	callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, struct WGPUStringView message, void* userdata1, [[maybe_unused]] void* userdata2) {
		UserData& userData = *reinterpret_cast<UserData*>(userdata1);
		if (status == WGPURequestAdapterStatus_Success) {
			userData.adapter = adapter;
		} else {
			std::cout << "Could not get WebGPU adapter: " << wgpu::StringView(message) << std::endl;
		}
		userData.requestEnded = true;
	};

	// Call to the WebGPU request adapter procedure
	wgpuInstanceRequestAdapter(instance, options, callbackInfo);

#endif // NOT __EMSCRIPTEN__

	assert(userData.requestEnded);

	return userData.adapter;
}

void inspectAdapter(WGPUAdapter adapter) {
#ifdef __EMSCRIPTEN__

	// Emscripten still uses an old version of the API

	WGPUSupportedLimits supportedLimits = {};
	supportedLimits.nextInChain = nullptr;

	bool success = wgpuAdapterGetLimits(adapter, &supportedLimits);

	if (success) {
		std::cout << "Adapter limits:" << std::endl;
		std::cout << " - maxTextureDimension1D: " << supportedLimits.limits.maxTextureDimension1D << std::endl;
		std::cout << " - maxTextureDimension2D: " << supportedLimits.limits.maxTextureDimension2D << std::endl;
		std::cout << " - maxTextureDimension3D: " << supportedLimits.limits.maxTextureDimension3D << std::endl;
		std::cout << " - maxTextureArrayLayers: " << supportedLimits.limits.maxTextureArrayLayers << std::endl;
	}

	size_t featureCount = wgpuAdapterEnumerateFeatures(adapter, nullptr);
	std::vector<WGPUFeatureName> features(featureCount);
	wgpuAdapterEnumerateFeatures(adapter, features.data());

	std::cout << "Adapter features:" << std::endl;
	std::cout << std::hex; // Write integers as hexadecimal to ease comparison with webgpu.h literals
	for (size_t i = 0; i < featureCount; ++i) {
		std::cout << " - 0x" << features[i] << std::endl;
	}
	std::cout << std::dec; // Restore decimal numbers

#else // __EMSCRIPTEN__
	WGPULimits supportedLimits = {};
	supportedLimits.nextInChain = nullptr;

#ifdef WEBGPU_BACKEND_DAWN
	bool success = wgpuAdapterGetLimits(adapter, &supportedLimits) == WGPUStatus_Success;
#else
	bool success = wgpuAdapterGetLimits(adapter, &supportedLimits);
#endif

	if (success) {
		std::cout << "Adapter limits:" << std::endl;
		std::cout << " - maxTextureDimension1D: " << supportedLimits.maxTextureDimension1D << std::endl;
		std::cout << " - maxTextureDimension2D: " << supportedLimits.maxTextureDimension2D << std::endl;
		std::cout << " - maxTextureDimension3D: " << supportedLimits.maxTextureDimension3D << std::endl;
		std::cout << " - maxTextureArrayLayers: " << supportedLimits.maxTextureArrayLayers << std::endl;
	}

	WGPUSupportedFeatures features;
	wgpuAdapterGetFeatures(adapter, &features);

	std::cout << "Adapter features:" << std::endl;
	std::cout << std::hex; // Write integers as hexadecimal to ease comparison with webgpu.h literals
	for (size_t i = 0; i < features.featureCount; ++i) {
		std::cout << " - 0x" << features.features[i] << std::endl;
	}
	std::cout << std::dec; // Restore decimal numbers

	wgpuSupportedFeaturesFreeMembers(features);

#endif // NOT __EMSCRIPTEN__

	WGPUAdapterInfo properties;
	properties.nextInChain = nullptr;
	wgpuAdapterGetInfo(adapter, &properties);
	std::cout << "Adapter properties:" << std::endl;
	std::cout << " - vendorID: " << properties.vendorID << std::endl;
	std::cout << " - vendorName: " << wgpu::StringView(properties.vendor) << std::endl;
	std::cout << " - architecture: " << wgpu::StringView(properties.architecture) << std::endl;
	std::cout << " - deviceID: " << properties.deviceID << std::endl;
	std::cout << " - name: " << wgpu::StringView(properties.device) << std::endl;
	std::cout << " - driverDescription: " << wgpu::StringView(properties.description) << std::endl;
	std::cout << std::hex;
	std::cout << " - adapterType: 0x" << properties.adapterType << std::endl;
	std::cout << " - backendType: 0x" << properties.backendType << std::endl;
	std::cout << std::dec; // Restore decimal numbers
	wgpuAdapterInfoFreeMembers(properties);
}

int main() {
	WGPUInstance instance = wgpuCreateInstance(nullptr);

	if (!instance) {
		std::cerr << "Could not initialize WebGPU!" << std::endl;
		return 1;
	}

	std::cout << "WGPU instance: " << instance << std::endl;

	std::cout << "Requesting adapter..." << std::endl;
	WGPURequestAdapterOptions adapterOpts = {};
	adapterOpts.nextInChain = nullptr;
	WGPUAdapter adapter = requestAdapterSync(instance, &adapterOpts);
	std::cout << "Got adapter: " << adapter << std::endl;

	// Display some information about the adapter
	inspectAdapter(adapter);

	// We no longer need to use the instance once we have the adapter
	wgpuInstanceRelease(instance);

	wgpuAdapterRelease(adapter);

	return 0;
}
