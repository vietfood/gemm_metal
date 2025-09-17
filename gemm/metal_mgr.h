#pragma once

#include <iostream>

#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLCounters.hpp"
#include "Metal/MTLDevice.hpp"

constexpr MTL::CounterSamplingPoint all_boundaries[] = {
    MTL::CounterSamplingPointAtStageBoundary,
    MTL::CounterSamplingPointAtDrawBoundary,
    MTL::CounterSamplingPointAtBlitBoundary,
    MTL::CounterSamplingPointAtDispatchBoundary,
    MTL::CounterSamplingPointAtTileDispatchBoundary};

// https://developer.apple.com/documentation/metal/confirming-which-counters-and-counter-sets-a-gpu-supports?language=objc
inline MTL::CounterSet* get_counter_set(NS::String* common_counter_name,
                                        MTL::Device* device)
{
  auto counter_set = device->counterSets();
  for (uint i = 0; i < counter_set->count(); i++) {
    auto counter = static_cast<MTL::CounterSet*>(counter_set->object(i));
    if (counter->name()->isEqualToString(common_counter_name)) {
      return counter;
    }
  }
  return nullptr;
}

// https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers?language=objc
// When run this, my GPU (Apple's M2) supports Stage Boundary only, life will be
// easier if it can support more, fk Apple
inline std::vector<size_t> sampling_boundaries_for(MTL::Device* device)
{
  std::vector<std::string> boundary_names = {
      "atStageBoundary", "atDrawBoundary", "atBlitBoundary",
      "atDispatchBoundary", "atTileDispatchBoundary"};

  std::cout << "The GPU device supports the following sampling boundary/ies: [";

  std::vector<size_t> boundaries;

  for (size_t index = 0; index < boundary_names.size(); index++) {
    if (device->supportsCounterSampling(all_boundaries[index])) {
      if (boundaries.size() >= 1) {
        std::cout << ",";
      }
      std::cout << boundary_names[index];
      boundaries.push_back(index);
    }
  }
  std::cout << "]\n";

  return boundaries;
}

struct MetalMgr
{
  MTL::Device* device = nullptr;
  MTL::CommandQueue* cmd_queue = nullptr;
  MTL::CounterSet* counter_set = nullptr;
  MTL::CounterSampleBufferDescriptor* sample_desc = nullptr;
  MTL::CounterSampleBuffer* counter_buffer = nullptr;

  MetalMgr();
  ~MetalMgr();
};
