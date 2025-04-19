#include <cassert>
#include <stdexcept>

#include "Foundation/NSError.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLCounters.hpp"
#include "Metal/MTLResource.hpp"
#include "metal_mgr.h"

MetalMgr::MetalMgr()
{
  device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    throw std::runtime_error("Cannot create device");
  }

  cmd_queue = device->newCommandQueue();
  if (cmd_queue == nullptr) {
    throw std::runtime_error("Cannot create common_queue");
  }

  // https://developer.apple.com/documentation/metal/creating-a-counter-sample-buffer-to-store-a-gpus-counter-data-during-a-pass?language=objc
  counter_set = get_counter_set(MTL::CommonCounterSetTimestamp, device);
  if (counter_set == nullptr) {
    throw std::runtime_error("Cannot create counter set");
  }

  sample_desc = MTL::CounterSampleBufferDescriptor::alloc()->init();
  sample_desc->setCounterSet(counter_set);
  sample_desc->setStorageMode(MTL::StorageModeShared);
  sample_desc->setSampleCount(2);

  NS::Error* error = nullptr;
  counter_buffer = device->newCounterSampleBuffer(sample_desc, &error);

  if (error != nullptr) {
    const char* msg = error->localizedDescription()->utf8String();
    throw std::runtime_error("Cannot create counter buffer because: "
                             + std::string(msg));
  }
}

MetalMgr::~MetalMgr()
{
  counter_buffer->release();
  sample_desc->release();
  counter_set->release();
  cmd_queue->release();
  device->release();
}