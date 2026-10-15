/*
    Experimental Vulkan GPU3D renderer for melonDS.

    This directly rasterizes DS polygons with Vulkan. The result is pipelined one frame behind
    and read back for the existing software 2D compositor, which keeps capture effects and both
    screen layouts compatible with the old core. Edge marking, fog and shadow-volume stencil are
    intentionally left to a later accuracy pass; ordinary opaque/translucent textured geometry
    is handled on the GPU.
*/

#include "GPU3D_Vulkan.h"

#include "Config.h"
#include "GPU.h"
#include "melonds_vulkan_frag_spv.h"
#include "melonds_vulkan_vert_spv.h"
#define XXH_STATIC_LINKING_ONLY
#include "xxhash/xxhash.h"

#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <unistd.h>
#if defined(__linux__)
#include <pthread.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace GPU3D {

namespace {
constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
// At 1x every polygon is drawn as one quad per scanline it covers (see AddPolygonSpans).
constexpr size_t MaxSpans = 96 * 1024;
constexpr size_t MaxVertices = MaxSpans * 4;
constexpr size_t MaxIndices = MaxSpans * 6;
// A polygon covers at most every scanline (and a flat one, one).
constexpr size_t MaxSpanRows = 256;
// The span cache's hash table: a power of two, four times the most polygons a frame has.
constexpr size_t SpanCacheSlots = 8192;
// A polygon's span cache key: six words, then eight for each vertex (see BuildPolygonSpans).
constexpr size_t SpanKeyWords = 6 + 8 * 10;

struct PushConstants {
    u32 WBuffer;
    u32 Translucent;
};

using TraceClock = std::chrono::steady_clock;

struct VulkanTraceStats {
    u64 Frames = 0;
    u64 EmptyFrames = 0;
    u64 Polygons = 0;
    u64 DrawCalls = 0;
    u64 Vertices = 0;
    u64 BuildNanoseconds = 0;
    u64 WaitNanoseconds = 0;
    u64 UploadNanoseconds = 0;
    u64 RecordSubmitNanoseconds = 0;
    u64 MaximumWaitNanoseconds = 0;
};

bool TraceEnabled = false;
VulkanTraceStats TraceStats;

u64 ElapsedNanoseconds(TraceClock::time_point start, TraceClock::time_point end) {
    return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void RecordTrace(u64 build, u64 wait, u64 upload, u64 recordSubmit,
                 size_t polygons, size_t drawCalls, size_t vertices, bool empty) {
    if (!TraceEnabled) return;
    ++TraceStats.Frames;
    TraceStats.EmptyFrames += empty ? 1 : 0;
    TraceStats.Polygons += polygons;
    TraceStats.DrawCalls += drawCalls;
    TraceStats.Vertices += vertices;
    TraceStats.BuildNanoseconds += build;
    TraceStats.WaitNanoseconds += wait;
    TraceStats.UploadNanoseconds += upload;
    TraceStats.RecordSubmitNanoseconds += recordSubmit;
    TraceStats.MaximumWaitNanoseconds = std::max(TraceStats.MaximumWaitNanoseconds, wait);
    if (TraceStats.Frames < 300) return;

    constexpr double NanosecondsToMilliseconds = 1.0 / 1000000.0;
    const double frameScale = 1.0 / static_cast<double>(TraceStats.Frames);
    __android_log_print(ANDROID_LOG_INFO, "MelonDSVulkan3D",
            "trace: frames=%llu empty=%llu polygons=%.1f draws=%.1f vertices=%.1f; "
            "build=%.3fms wait=%.3fms upload=%.3fms record+submit=%.3fms maxWait=%.3fms",
            static_cast<unsigned long long>(TraceStats.Frames),
            static_cast<unsigned long long>(TraceStats.EmptyFrames),
            static_cast<double>(TraceStats.Polygons) * frameScale,
            static_cast<double>(TraceStats.DrawCalls) * frameScale,
            static_cast<double>(TraceStats.Vertices) * frameScale,
            static_cast<double>(TraceStats.BuildNanoseconds) * frameScale
                    * NanosecondsToMilliseconds,
            static_cast<double>(TraceStats.WaitNanoseconds) * frameScale
                    * NanosecondsToMilliseconds,
            static_cast<double>(TraceStats.UploadNanoseconds) * frameScale
                    * NanosecondsToMilliseconds,
            static_cast<double>(TraceStats.RecordSubmitNanoseconds) * frameScale
                    * NanosecondsToMilliseconds,
            static_cast<double>(TraceStats.MaximumWaitNanoseconds) * NanosecondsToMilliseconds);
    TraceStats = {};
}

// Polygon attributes the fragment shader reads (flat): blend mode, polygon ID, and in bits
// 8 and 9 whether it faces the viewer and is W-buffered.
u32 PolygonAttr(const Polygon* polygon) {
    u32 attr = polygon->Attr & 0x1f00c8f0;
    if (polygon->FacingView) attr |= 1 << 8;
    if (polygon->WBuffer) attr |= 1 << 9;
    return attr;
}

template <u32 Size>
void CopyDirtyRegions(u8* destination, const u8* source,
                      NonStupidBitField<Size>& dirty) {
    auto it = dirty.Begin();
    if (it == dirty.End()) return;

    u32 first = *it;
    u32 last = first;
    ++it;
    while (it != dirty.End()) {
        const u32 current = *it;
        if (current != last + 1) {
            const size_t offset = static_cast<size_t>(first) * GPU::VRAMDirtyGranularity;
            const size_t bytes = static_cast<size_t>(last - first + 1)
                    * GPU::VRAMDirtyGranularity;
            std::memcpy(destination + offset, source + offset, bytes);
            first = current;
        }
        last = current;
        ++it;
    }
    const size_t offset = static_cast<size_t>(first) * GPU::VRAMDirtyGranularity;
    const size_t bytes = static_cast<size_t>(last - first + 1) * GPU::VRAMDirtyGranularity;
    std::memcpy(destination + offset, source + offset, bytes);
}
}

VulkanRenderer::VulkanRenderer() : Renderer3D(false) {}

VulkanRenderer::~VulkanRenderer() {
    DeInit();
}

bool VulkanRenderer::ExportSharedContext(MelonDSVulkanSharedContext& context) const {
    if (!Ready || Instance == VK_NULL_HANDLE || PhysicalDevice == VK_NULL_HANDLE
            || Device == VK_NULL_HANDLE || Queue == VK_NULL_HANDLE) return false;
    context = {};
    context.version = MELONDS_VULKAN_SHARED_CONTEXT_VERSION;
    context.instance = Instance;
    context.physicalDevice = PhysicalDevice;
    context.device = Device;
    context.queue = Queue;
    context.queueFamily = QueueFamily;
    return true;
}

bool VulkanRenderer::Fail(const char* message) {
    __android_log_print(ANDROID_LOG_ERROR, "MelonDSVulkan3D", "%s", message);
    return false;
}

u32 VulkanRenderer::FindMemoryType(u32 bits, VkMemoryPropertyFlags required,
                                    VkMemoryPropertyFlags preferred) const {
    u32 fallback = ~0u;
    for (u32 i = 0; i < MemoryProperties.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        const auto flags = MemoryProperties.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if ((flags & preferred) == preferred) return i;
        if (fallback == ~0u) fallback = i;
    }
    return fallback;
}

bool VulkanRenderer::CreateDevice() {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "melonDS Vulkan GPU3D";
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &application;
#if defined(__ANDROID__)
    // The Android frame presenter can share this device, so it gets the surface extensions.
    // Elsewhere (a Linux build, for testing) the renderer only draws offscreen.
    const char* instanceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    };
    instanceInfo.enabledExtensionCount = 2;
    instanceInfo.ppEnabledExtensionNames = instanceExtensions;
#endif
    if (vkCreateInstance(&instanceInfo, nullptr, &Instance) != VK_SUCCESS)
        return Fail("could not create instance");

    u32 count = 0;
    vkEnumeratePhysicalDevices(Instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(Instance, &count, devices.data());
    for (auto candidate : devices) {
        u32 familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
        for (u32 family = 0; family < familyCount; ++family) {
            if (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                PhysicalDevice = candidate;
                QueueFamily = family;
                break;
            }
        }
        if (PhysicalDevice != VK_NULL_HANDLE) break;
    }
    if (PhysicalDevice == VK_NULL_HANDLE) return Fail("no graphics queue");
    vkGetPhysicalDeviceMemoryProperties(PhysicalDevice, &MemoryProperties);
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(PhysicalDevice, &deviceProperties);
    __android_log_print(ANDROID_LOG_INFO, "MelonDSVulkan3D",
            "GPU3D device: %s, vendor=%04x, device=%04x, driver=%u, api=%u.%u.%u",
            deviceProperties.deviceName, deviceProperties.vendorID, deviceProperties.deviceID,
            deviceProperties.driverVersion, VK_VERSION_MAJOR(deviceProperties.apiVersion),
            VK_VERSION_MINOR(deviceProperties.apiVersion),
            VK_VERSION_PATCH(deviceProperties.apiVersion));

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = QueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(PhysicalDevice, &supported);
    VkPhysicalDeviceFeatures enabled{};
    enabled.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
#if defined(__ANDROID__)
    const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;
#endif
    if (vkCreateDevice(PhysicalDevice, &deviceInfo, nullptr, &Device) != VK_SUCCESS)
        return Fail("could not create device");
    vkGetDeviceQueue(Device, QueueFamily, 0, &Queue);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = QueueFamily;
    if (vkCreateCommandPool(Device, &poolInfo, nullptr, &CommandPool) != VK_SUCCESS)
        return Fail("could not create command pool");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = CommandPool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(Device, &allocate, &CommandBuffer) != VK_SUCCESS)
        return Fail("could not allocate command buffer");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(Device, &fenceInfo, nullptr, &Fence) != VK_SUCCESS)
        return Fail("could not create fence");
    return true;
}

bool VulkanRenderer::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result,
                                  bool readback) {
    result.Size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(Device, &info, nullptr, &result.Handle) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(Device, result.Handle, &requirements);
    // The CPU reads the readback buffers pixel by pixel (GetLine), so they want cached memory:
    // uncached reads of the mapping made GetLine a quarter of the core's CPU time. Upload
    // buffers are only written and keep the usual (write-combined) coherent memory.
    u32 type = readback ? FindMemoryType(requirements.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                 | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        : ~0u;
    if (type == ~0u)
        type = FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == ~0u) return false;
    result.Coherent = (MemoryProperties.memoryTypes[type].propertyFlags
            & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    if (vkAllocateMemory(Device, &allocation, nullptr, &result.Memory) != VK_SUCCESS
            || vkBindBufferMemory(Device, result.Handle, result.Memory, 0) != VK_SUCCESS
            || vkMapMemory(Device, result.Memory, 0, size, 0, &result.Mapped) != VK_SUCCESS)
        return false;
    return true;
}

void VulkanRenderer::DestroyBuffer(Buffer& buffer) {
    if (buffer.Mapped) vkUnmapMemory(Device, buffer.Memory);
    if (buffer.Handle) vkDestroyBuffer(Device, buffer.Handle, nullptr);
    if (buffer.Memory) vkFreeMemory(Device, buffer.Memory, nullptr);
    buffer = {};
}

void VulkanRenderer::Flush(Buffer& buffer) {
    if (buffer.Coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.Memory;
    range.size = VK_WHOLE_SIZE;
    vkFlushMappedMemoryRanges(Device, 1, &range);
}

void VulkanRenderer::Invalidate(Buffer& buffer) {
    if (buffer.Coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.Memory;
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(Device, 1, &range);
}

bool VulkanRenderer::CreateShader(const u8* bytes, size_t size, VkShaderModule& result) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = size;
    info.pCode = reinterpret_cast<const u32*>(bytes);
    return vkCreateShaderModule(Device, &info, nullptr, &result) == VK_SUCCESS;
}

bool VulkanRenderer::CreatePipelineCache() {
    PipelineCachePath.clear();
    PipelineCacheDiskData.clear();
    const char* configuredPath = std::getenv("MELONDS_VULKAN_PIPELINE_CACHE");
    if (configuredPath != nullptr && configuredPath[0] != '\0') {
        PipelineCachePath = configuredPath;
        const int file = open(PipelineCachePath.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat fileInfo{};
        if (file >= 0 && fstat(file, &fileInfo) == 0
                && fileInfo.st_size > 0 && fileInfo.st_size <= 64L * 1024L * 1024L) {
            PipelineCacheDiskData.resize(static_cast<size_t>(fileInfo.st_size));
            size_t offset = 0;
            while (offset < PipelineCacheDiskData.size()) {
                const ssize_t bytes = read(file, PipelineCacheDiskData.data() + offset,
                                           PipelineCacheDiskData.size() - offset);
                if (bytes <= 0) {
                    PipelineCacheDiskData.clear();
                    break;
                }
                offset += static_cast<size_t>(bytes);
            }
        }
        if (file >= 0) close(file);
    }

    // Reject stale data before giving it to the driver. The Vulkan header ties
    // an opaque cache blob to the exact vendor, device, and pipeline-cache UUID.
    if (!PipelineCacheDiskData.empty()) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(PhysicalDevice, &properties);
        constexpr size_t HeaderSize = sizeof(u32) * 4 + VK_UUID_SIZE;
        u32 headerLength = 0;
        u32 headerVersion = 0;
        u32 vendor = 0;
        u32 device = 0;
        if (PipelineCacheDiskData.size() >= HeaderSize) {
            std::memcpy(&headerLength, PipelineCacheDiskData.data(), sizeof(u32));
            std::memcpy(&headerVersion, PipelineCacheDiskData.data() + 4, sizeof(u32));
            std::memcpy(&vendor, PipelineCacheDiskData.data() + 8, sizeof(u32));
            std::memcpy(&device, PipelineCacheDiskData.data() + 12, sizeof(u32));
        }
        const bool compatible = PipelineCacheDiskData.size() >= HeaderSize
                && headerLength >= HeaderSize
                && headerLength <= PipelineCacheDiskData.size()
                && headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE
                && vendor == properties.vendorID
                && device == properties.deviceID
                && std::memcmp(PipelineCacheDiskData.data() + 16,
                               properties.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        if (!compatible) {
            __android_log_print(ANDROID_LOG_WARN, "MelonDSVulkan3D",
                    "Ignoring incompatible pipeline cache: %s", PipelineCachePath.c_str());
            PipelineCacheDiskData.clear();
        }
    }

    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = PipelineCacheDiskData.size();
    info.pInitialData = PipelineCacheDiskData.empty() ? nullptr : PipelineCacheDiskData.data();
    VkResult result = vkCreatePipelineCache(Device, &info, nullptr, &PipelineCache);
    if (result != VK_SUCCESS && !PipelineCacheDiskData.empty()) {
        PipelineCacheDiskData.clear();
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        result = vkCreatePipelineCache(Device, &info, nullptr, &PipelineCache);
    }
    if (result != VK_SUCCESS) return Fail("could not create pipeline cache");
    __android_log_print(ANDROID_LOG_INFO, "MelonDSVulkan3D",
            "Pipeline cache: %s (%zu bytes), path=%s",
            PipelineCacheDiskData.empty() ? "cold" : "loaded",
            PipelineCacheDiskData.size(),
            PipelineCachePath.empty() ? "disabled" : PipelineCachePath.c_str());
    return true;
}

void VulkanRenderer::SavePipelineCache() {
    if (PipelineCache == VK_NULL_HANDLE || PipelineCachePath.empty()) return;
    size_t size = 0;
    if (vkGetPipelineCacheData(Device, PipelineCache, &size, nullptr) != VK_SUCCESS
            || size == 0 || size > 64u * 1024u * 1024u) return;
    std::vector<u8> data(size);
    if (vkGetPipelineCacheData(Device, PipelineCache, &size, data.data()) != VK_SUCCESS) return;
    data.resize(size);
    if (data == PipelineCacheDiskData) return;

    const std::string temporaryPath = PipelineCachePath + ".tmp";
    const int file = open(temporaryPath.c_str(),
                          O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (file < 0) {
        __android_log_print(ANDROID_LOG_WARN, "MelonDSVulkan3D",
                "Could not open pipeline-cache temporary file: %s", temporaryPath.c_str());
        return;
    }
    size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t bytes = write(file, data.data() + offset, data.size() - offset);
        if (bytes <= 0) break;
        offset += static_cast<size_t>(bytes);
    }
    const bool wrote = offset == data.size() && fsync(file) == 0;
    const bool closed = close(file) == 0;
    if (!wrote || !closed || rename(temporaryPath.c_str(), PipelineCachePath.c_str()) != 0) {
        unlink(temporaryPath.c_str());
        __android_log_print(ANDROID_LOG_WARN, "MelonDSVulkan3D",
                "Could not persist pipeline cache: %s", PipelineCachePath.c_str());
        return;
    }
    PipelineCacheDiskData = std::move(data);
    __android_log_print(ANDROID_LOG_INFO, "MelonDSVulkan3D",
            "Saved pipeline cache: %zu bytes to %s",
            PipelineCacheDiskData.size(), PipelineCachePath.c_str());
}

bool VulkanRenderer::CreateTargets() {
    Width = 256u * static_cast<u32>(Scale);
    Height = 192u * static_cast<u32>(Scale);
    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {Width, Height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(Device, &info, nullptr, &image) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(Device, image, &requirements);
        u32 type = FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == ~0u) return false;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (vkAllocateMemory(Device, &allocation, nullptr, &memory) != VK_SUCCESS
                || vkBindImageMemory(Device, image, memory, 0) != VK_SUCCESS) return false;
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        return vkCreateImageView(Device, &viewInfo, nullptr, &view) == VK_SUCCESS;
    };

    if (!createImage(ColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, ColorImage, ColorMemory, ColorView))
        return Fail("could not create color target");
    // A float depth buffer holds the DS's 24-bit depth values exactly.
    for (VkFormat candidate : {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
                               VK_FORMAT_D16_UNORM_S8_UINT}) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(PhysicalDevice, candidate, &properties);
        if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            DepthFormat = candidate;
            break;
        }
    }
    if (DepthFormat == VK_FORMAT_UNDEFINED || !createImage(DepthFormat,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
            DepthImage, DepthMemory, DepthView)) return Fail("could not create depth target");

    VkAttachmentDescription attachments[2]{};
    attachments[0].format = ColorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    attachments[1].format = DepthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo renderInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderInfo.attachmentCount = 2;
    renderInfo.pAttachments = attachments;
    renderInfo.subpassCount = 1;
    renderInfo.pSubpasses = &subpass;
    renderInfo.dependencyCount = 1;
    renderInfo.pDependencies = &dependency;
    if (vkCreateRenderPass(Device, &renderInfo, nullptr, &RenderPass) != VK_SUCCESS)
        return Fail("could not create render pass");
    VkImageView views[] = {ColorView, DepthView};
    VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.renderPass = RenderPass;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = views;
    fbInfo.width = Width;
    fbInfo.height = Height;
    fbInfo.layers = 1;
    if (vkCreateFramebuffer(Device, &fbInfo, nullptr, &FramebufferObject) != VK_SUCCESS)
        return Fail("could not create framebuffer");
    const VkDeviceSize readbackSize = static_cast<VkDeviceSize>(Width) * Height * 4;
    return CreateBuffer(readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, ReadbackBuffers[0], true)
            && CreateBuffer(readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, ReadbackBuffers[1], true);
}

void VulkanRenderer::DestroyTargets() {
    for (auto& pipeline : Pipelines) {
        if (pipeline) vkDestroyPipeline(Device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (FramebufferObject) vkDestroyFramebuffer(Device, FramebufferObject, nullptr);
    if (RenderPass) vkDestroyRenderPass(Device, RenderPass, nullptr);
    if (ColorView) vkDestroyImageView(Device, ColorView, nullptr);
    if (DepthView) vkDestroyImageView(Device, DepthView, nullptr);
    if (ColorImage) vkDestroyImage(Device, ColorImage, nullptr);
    if (DepthImage) vkDestroyImage(Device, DepthImage, nullptr);
    if (ColorMemory) vkFreeMemory(Device, ColorMemory, nullptr);
    if (DepthMemory) vkFreeMemory(Device, DepthMemory, nullptr);
    for (auto& readback : ReadbackBuffers) DestroyBuffer(readback);
    ReadbackOutput = RawOutput.data();
    ReadbackWriteIndex = 0;
    OutputCurrent = false;
    FramebufferObject = VK_NULL_HANDLE;
    RenderPass = VK_NULL_HANDLE;
    ColorView = DepthView = VK_NULL_HANDLE;
    ColorImage = DepthImage = VK_NULL_HANDLE;
    ColorMemory = DepthMemory = VK_NULL_HANDLE;
}

bool VulkanRenderer::CreateDescriptors() {
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(Device, &layoutInfo, nullptr, &DescriptorLayout) != VK_SUCCESS)
        return false;
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineInfo.setLayoutCount = 1;
    pipelineInfo.pSetLayouts = &DescriptorLayout;
    pipelineInfo.pushConstantRangeCount = 1;
    pipelineInfo.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(Device, &pipelineInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
        return false;
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(Device, &poolInfo, nullptr, &DescriptorPool) != VK_SUCCESS)
        return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = DescriptorPool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &DescriptorLayout;
    if (vkAllocateDescriptorSets(Device, &allocate, &DescriptorSet) != VK_SUCCESS) return false;
    VkDescriptorBufferInfo infos[] = {{UniformBuffer.Handle, 0, UniformBuffer.Size},
                                      {TextureBuffer.Handle, 0, TextureBuffer.Size},
                                      {PaletteBuffer.Handle, 0, PaletteBuffer.Size}};
    VkWriteDescriptorSet writes[3]{};
    for (u32 i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = DescriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                           : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(Device, 3, writes, 0, nullptr);
    return true;
}

bool VulkanRenderer::CreatePipeline(bool translucent, bool depthEqual, bool depthWrite, bool line,
                                    VkPipeline& result) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = VertexShader;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = FragmentShader;
    stages[1].pName = "main";
    VkVertexInputBindingDescription binding{0, sizeof(VertexGPU), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R16G16_UINT, 0},
        {1, 0, VK_FORMAT_R8G8B8A8_UINT, 12},
        {2, 0, VK_FORMAT_R16G16_SINT, 16},
        {3, 0, VK_FORMAT_R32G32B32_UINT, 20},
        {4, 0, VK_FORMAT_R32G32_UINT, 4},
        {5, 0, VK_FORMAT_R32G32B32_UINT, 32},
    };
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 6;
    vertexInput.pVertexAttributeDescriptions = attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = line ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = depthEqual ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS;
    // The stencil holds what the DS keeps in its attribute buffer for translucency: bit 6 set
    // when a translucent pixel was drawn there, bits 0-5 that polygon's ID. A translucent
    // polygon does not blend over a pixel already blended by a translucent polygon with the
    // same ID (games draw shadows and other overlays as several overlapping pieces that share
    // one ID); an opaque pixel clears it. The reference is set per draw (dynamic state).
    depth.stencilTestEnable = VK_TRUE;
    depth.front.failOp = VK_STENCIL_OP_KEEP;
    depth.front.passOp = VK_STENCIL_OP_REPLACE;
    depth.front.depthFailOp = VK_STENCIL_OP_KEEP;
    depth.front.compareOp = translucent ? VK_COMPARE_OP_NOT_EQUAL : VK_COMPARE_OP_ALWAYS;
    depth.front.compareMask = 0x7f;
    depth.front.writeMask = 0x7f;
    depth.back = depth.front;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 0xf;
    blend.blendEnable = translucent ? VK_TRUE : VK_FALSE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.alphaBlendOp = VK_BLEND_OP_MAX;
    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 3;
    dynamic.pDynamicStates = dynamicStates;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = &dynamic;
    info.layout = PipelineLayout;
    info.renderPass = RenderPass;
    return vkCreateGraphicsPipelines(Device, PipelineCache, 1, &info, nullptr, &result)
            == VK_SUCCESS;
}

bool VulkanRenderer::CreatePipelines() {
    if (VertexShader == VK_NULL_HANDLE
            && !CreateShader(melonds_vulkan_vert_spv, melonds_vulkan_vert_spv_size, VertexShader))
        return Fail("could not create vertex shader");
    if (FragmentShader == VK_NULL_HANDLE
            && !CreateShader(melonds_vulkan_frag_spv, melonds_vulkan_frag_spv_size,
                             FragmentShader))
        return Fail("could not create fragment shader");
    for (u32 key = 0; key < Pipelines.size(); ++key) {
        if (!CreatePipeline(key & 1, key & 2, key & 4, key & 8, Pipelines[key]))
            return Fail("could not create graphics pipeline");
    }
    SavePipelineCache();
    return true;
}

bool VulkanRenderer::Init() {
    char traceProperty[PROP_VALUE_MAX]{};
    __system_property_get("debug.multigame.melonds_vulkan_trace", traceProperty);
    TraceEnabled = traceProperty[0] == '1';
    TraceStats = {};
    if (!CreateDevice()) return false;
    if (!CreatePipelineCache()) return false;
    if (!CreateBuffer(sizeof(ShaderConfig), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, UniformBuffer)
            || !CreateBuffer(512 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, TextureBuffer)
            || !CreateBuffer(128 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, PaletteBuffer)
            || !CreateBuffer(MaxVertices * sizeof(VertexGPU), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VertexBuffer)
            || !CreateBuffer(MaxIndices * sizeof(u32), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             IndexBuffer)
            || !CreateBuffer(MaxSpanRows * 6 * sizeof(u16), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             SpanIndexBuffer)) return Fail("could not allocate upload buffers");
    auto* spanIndices = static_cast<u16*>(SpanIndexBuffer.Mapped);
    // Vertices 0 and 1 of a span are the top and bottom of its first pixel, 2 and 3 of its last.
    for (u32 span = 0; span < MaxSpanRows; ++span)
        for (u32 index : {0u, 2u, 1u, 1u, 2u, 3u}) *spanIndices++ = static_cast<u16>(span * 4 + index);
    Flush(SpanIndexBuffer);
    // Left uninitialized: only the part a frame writes is ever touched.
    for (auto& vertices : SpanVertices) vertices.reset(new VertexGPU[MaxVertices]);
    for (auto& cache : SpanCaches) {
        cache.Keys.clear();
        cache.Keys.reserve(2048 * SpanKeyWords);
        cache.Entries.clear();
        cache.Entries.reserve(2048);
        // Slots of an older generation are empty, so a cache that was never filled is empty.
        cache.Slots.assign(SpanCacheSlots, SpanCacheSlot{0, 0});
        cache.Generation = 1;
        cache.Vertices = 0;
    }
    SpanVerticesUploaded = false;
    if (!CreateTargets() || !CreateDescriptors() || !CreatePipelines()) return false;
    Draws.reserve(2048);
    Ready = true;
    EmptyFramePending = false;
    Worker = std::thread([this] { WorkerMain(); });
    __android_log_print(ANDROID_LOG_INFO, "MelonDSVulkan3D",
            "Initialized %ux%u with mapped ping-pong readback", Width, Height);
    return true;
}

void VulkanRenderer::DeInit() {
    StopWorker();
    if (!Device) return;
    vkDeviceWaitIdle(Device);
    SavePipelineCache();
    DestroyTargets();
    DestroyBuffer(UniformBuffer);
    DestroyBuffer(TextureBuffer);
    DestroyBuffer(PaletteBuffer);
    DestroyBuffer(VertexBuffer);
    DestroyBuffer(IndexBuffer);
    DestroyBuffer(SpanIndexBuffer);
    for (auto& vertices : SpanVertices) vertices.reset();
    SpanVerticesUploaded = false;
    if (VertexShader) vkDestroyShaderModule(Device, VertexShader, nullptr);
    if (FragmentShader) vkDestroyShaderModule(Device, FragmentShader, nullptr);
    if (DescriptorPool) vkDestroyDescriptorPool(Device, DescriptorPool, nullptr);
    if (PipelineCache) vkDestroyPipelineCache(Device, PipelineCache, nullptr);
    if (PipelineLayout) vkDestroyPipelineLayout(Device, PipelineLayout, nullptr);
    if (DescriptorLayout) vkDestroyDescriptorSetLayout(Device, DescriptorLayout, nullptr);
    if (Fence) vkDestroyFence(Device, Fence, nullptr);
    if (CommandPool) vkDestroyCommandPool(Device, CommandPool, nullptr);
    vkDestroyDevice(Device, nullptr);
    if (Instance) vkDestroyInstance(Instance, nullptr);
    Device = VK_NULL_HANDLE;
    Instance = VK_NULL_HANDLE;
    PipelineCache = VK_NULL_HANDLE;
    Ready = false;
    SubmissionPending = false;
}

void VulkanRenderer::Reset() {
    FinishRendering();
    EmptyFramePending = false;
    if (Device && SubmissionPending) {
        vkWaitForFences(Device, 1, &Fence, VK_TRUE, UINT64_MAX);
        SubmissionPending = false;
    }
    RawOutput.fill(0);
    Output.fill(0);
    ReadbackOutput = RawOutput.data();
    ReadbackWriteIndex = 0;
    OutputCurrent = false;
}

void VulkanRenderer::SetRenderSettings(GPU::RenderSettings& settings) {
    FinishRendering();
    OutputCurrent = false;
    BetterPolygons = settings.GL_BetterPolygons;
    const int requested = std::clamp(settings.GL_ScaleFactor, 1, 4);
    if (requested == Scale || !Ready) return;
    vkDeviceWaitIdle(Device);
    SubmissionPending = false;
    EmptyFramePending = false;
    DestroyTargets();
    Scale = requested;
    if (!CreateTargets() || !CreatePipelines()) {
        Ready = false;
        Fail("could not resize render targets");
    }
}

bool VulkanRenderer::UploadMemory() {
    auto* texture = static_cast<u8*>(TextureBuffer.Mapped);
    auto* palette = static_cast<u8*>(PaletteBuffer.Mapped);
    auto textureDirty = GPU::VRAMDirty_Texture.DeriveState(GPU::VRAMMap_Texture);
    auto paletteDirty = GPU::VRAMDirty_TexPal.DeriveState(GPU::VRAMMap_TexPal);
    const bool textureChanged = GPU::MakeVRAMFlat_TextureCoherent(textureDirty);
    const bool paletteChanged = GPU::MakeVRAMFlat_TexPalCoherent(paletteDirty);

    // Upload only the 512-byte regions written or remapped since the previous 3D frame.
    // Static textures are common and copying all 640 KiB every frame showed up directly in
    // the Shield profile. The core's VRAM tracker also preserves the correct OR behavior if
    // software maps more than one physical bank over the same texture address.
    CopyDirtyRegions(texture, GPU::VRAMFlat_Texture, textureDirty);
    CopyDirtyRegions(palette, GPU::VRAMFlat_TexPal, paletteDirty);
    if (textureChanged) Flush(TextureBuffer);
    if (paletteChanged) Flush(PaletteBuffer);
    return textureChanged || paletteChanged;
}

u32 VulkanRenderer::AddVertex(Polygon* polygon, u32 index) {
    Vertex* vertex = polygon->Vertices[index];
    const u32 z = polygon->FinalZ[index];
    const u32 w = polygon->FinalW[index];
    const u32 x = (Scale > 1 ? (vertex->HiresPosition[0] * Scale) >> 4 : vertex->FinalPosition[0]) << 4;
    const u32 y = (Scale > 1 ? (vertex->HiresPosition[1] * Scale) >> 4 : vertex->FinalPosition[1]) << 4;
    const u32 attr = PolygonAttr(polygon);
    const u32 alpha = (polygon->Attr >> 16) & 31;
    VertexOut[VertexCount++] = {x | (y << 16), z, w,
        static_cast<u32>(vertex->FinalColor[0] >> 1)
                | (static_cast<u32>(vertex->FinalColor[1] >> 1) << 8)
                | (static_cast<u32>(vertex->FinalColor[2] >> 1) << 16) | (alpha << 24),
        static_cast<u16>(vertex->TexCoords[0])
                | (static_cast<u32>(static_cast<u16>(vertex->TexCoords[1])) << 16),
        attr, polygon->TexParam, polygon->TexPalette, 0, 0, 0};
    return VertexCount - 1;
}

// At 1x, polygons are rasterized the way the DS (and melonDS's software renderer) does it:
// the edges are stepped scanline by scanline with SoftRenderer's own Slope and Interpolator,
// and each scanline's span becomes one quad. The GPU then only fills spans, interpolating
// across them, so every polygon covers exactly the pixels it covers on the DS. Drawing the
// polygons as plain triangles instead covers fewer pixels along their edges, because the DS
// fills whole edge pixels (all of them when edge marking or antialiasing is on, as in the
// Pokémon games); where meshes meet with T-junctions or vertices rounded apart, that left
// one-pixel cracks of whatever lies behind. The quads also put the attributes where the DS
// samples them: values are interpolated along the edges per scanline exactly as on the DS,
// and the quad is shifted half a pixel so that the GPU's pixel centers fall on the DS's
// integer sample positions. Each span's four vertices are the quad's top and bottom at its
// first and last pixel; they are drawn with SpanIndexBuffer's pattern. BuildPolygonSpans has
// checked that the polygon is drawn and that its rows fit.
void VulkanRenderer::AddPolygonSpans(Polygon* polygon) {
    const u32 alpha = (polygon->Attr >> 16) & 31;
    const s32 ytop = polygon->YTop;
    const s32 ybottom = polygon->YBottom == ytop ? ytop + 1 : polygon->YBottom;

    using Soft = SoftRenderer;
    Soft::RendererPolygon rp{};
    const u32 count = polygon->NumVertices;
    auto vertexX = [&](u32 index) { return polygon->Vertices[index]->FinalPosition[0]; };
    auto vertexY = [&](u32 index) { return polygon->Vertices[index]->FinalPosition[1]; };
    // SoftRenderer::SetupPolygonLeftEdge/RightEdge.
    auto setupLeft = [&](s32 y) {
        while (y >= vertexY(rp.NextVL) && rp.CurVL != polygon->VBottom) {
            rp.CurVL = rp.NextVL;
            rp.NextVL = polygon->FacingView ? (rp.CurVL + 1 >= count ? 0 : rp.CurVL + 1)
                                            : (rp.CurVL == 0 ? count - 1 : rp.CurVL - 1);
        }
        rp.XL = rp.SlopeL.Setup(vertexX(rp.CurVL), vertexX(rp.NextVL), vertexY(rp.CurVL),
                                vertexY(rp.NextVL), polygon->FinalW[rp.CurVL],
                                polygon->FinalW[rp.NextVL], y);
    };
    auto setupRight = [&](s32 y) {
        while (y >= vertexY(rp.NextVR) && rp.CurVR != polygon->VBottom) {
            rp.CurVR = rp.NextVR;
            rp.NextVR = polygon->FacingView ? (rp.CurVR == 0 ? count - 1 : rp.CurVR - 1)
                                            : (rp.CurVR + 1 >= count ? 0 : rp.CurVR + 1);
        }
        rp.XR = rp.SlopeR.Setup(vertexX(rp.CurVR), vertexX(rp.NextVR), vertexY(rp.CurVR),
                                vertexY(rp.NextVR), polygon->FinalW[rp.CurVR],
                                polygon->FinalW[rp.NextVR], y);
    };

    // SoftRenderer::SetupPolygon.
    rp.PolyData = polygon;
    rp.CurVL = rp.CurVR = polygon->VTop;
    if (polygon->FacingView) {
        rp.NextVL = rp.CurVL + 1 >= count ? 0 : rp.CurVL + 1;
        rp.NextVR = rp.CurVR == 0 ? count - 1 : rp.CurVR - 1;
    } else {
        rp.NextVL = rp.CurVL == 0 ? count - 1 : rp.CurVL - 1;
        rp.NextVR = rp.CurVR + 1 >= count ? 0 : rp.CurVR + 1;
    }
    const bool flat = polygon->YBottom == polygon->YTop;
    if (flat) {
        u32 left = 0;
        u32 right = 0;
        for (u32 i : {1u, count - 1}) {
            if (vertexX(i) < vertexX(left)) left = i;
            if (vertexX(i) > vertexX(right)) right = i;
        }
        rp.CurVL = rp.NextVL = left;
        rp.CurVR = rp.NextVR = right;
        rp.XL = rp.SlopeL.SetupDummy(vertexX(left));
        rp.XR = rp.SlopeR.SetupDummy(vertexX(right));
    } else {
        setupLeft(ytop);
        setupRight(ytop);
    }

    const u32 attr = PolygonAttr(polygon);
    const bool wbuffer = polygon->WBuffer;
    // Edges are always filled with edge marking or antialiasing; otherwise by slope.
    const bool fillAllEdges = (RenderDispCnt & ((1 << 4) | (1 << 5))) != 0;
    for (s32 y = ytop; y < ybottom; ++y) {
        // SoftRenderer::RenderPolygonScanline.
        if (!flat) {
            if (y >= vertexY(rp.NextVL) && rp.CurVL != polygon->VBottom) setupLeft(y);
            if (y >= vertexY(rp.NextVR) && rp.CurVR != polygon->VBottom) setupRight(y);
        }
        bool fillLeft = true;
        bool fillRight = true;
        if (!fillAllEdges) {
            fillLeft = rp.SlopeL.Negative || !rp.SlopeL.XMajor;
            fillRight = (!rp.SlopeR.Negative && rp.SlopeR.XMajor) || rp.SlopeR.Increment == 0;
        }
        s32 wl = rp.SlopeL.Interp.Interpolate(polygon->FinalW[rp.CurVL], polygon->FinalW[rp.NextVL]);
        s32 wr = rp.SlopeR.Interp.Interpolate(polygon->FinalW[rp.CurVR], polygon->FinalW[rp.NextVR]);
        s32 zl = rp.SlopeL.Interp.InterpolateZ(polygon->FinalZ[rp.CurVL], polygon->FinalZ[rp.NextVL], wbuffer);
        s32 zr = rp.SlopeR.Interp.InterpolateZ(polygon->FinalZ[rp.CurVR], polygon->FinalZ[rp.NextVR], wbuffer);
        s32 xstart = rp.XL;
        s32 xend = rp.XR;
        Vertex* leftCurrent;
        Vertex* leftNext;
        Vertex* rightCurrent;
        Vertex* rightNext;
        Soft::Interpolator<1>* interpStart;
        Soft::Interpolator<1>* interpEnd;
        s32 leftLength = 1, rightLength = 1, coverage;
        if (xstart > xend) {
            leftCurrent = polygon->Vertices[rp.CurVR];
            leftNext = polygon->Vertices[rp.NextVR];
            rightCurrent = polygon->Vertices[rp.CurVL];
            rightNext = polygon->Vertices[rp.NextVL];
            interpStart = &rp.SlopeR.Interp;
            interpEnd = &rp.SlopeL.Interp;
            std::swap(xstart, xend);
            std::swap(wl, wr);
            std::swap(zl, zr);
            std::swap(fillLeft, fillRight);
        } else {
            leftCurrent = polygon->Vertices[rp.CurVL];
            leftNext = polygon->Vertices[rp.NextVL];
            rightCurrent = polygon->Vertices[rp.CurVR];
            rightNext = polygon->Vertices[rp.NextVR];
            interpStart = &rp.SlopeL.Interp;
            interpEnd = &rp.SlopeR.Interp;
            // Edge lengths only matter for an edge that is not filled.
            if (!fillLeft || !fillRight) {
                rp.SlopeL.EdgeParams(&leftLength, &coverage);
                rp.SlopeR.EdgeParams(&rightLength, &coverage);
            }
        }

        // The pixels the three parts of the software renderer's scanline loop visit.
        s32 x = std::max(xstart, 0);
        s32 first = -1;
        s32 last = -1;
        const s32 leftLimit = std::min({xstart + leftLength, xend + 1, 256});
        if (!fillLeft) {
            x = std::min(leftLimit, xend - rightLength + 1);
        } else if (x < leftLimit) {
            first = x;
            last = leftLimit;
            x = leftLimit;
        }
        const s32 innerLimit = std::min({xend - rightLength + 1, xend + 1, 256});
        if (x < innerLimit) {
            if (first < 0) first = x;
            last = innerLimit;
            x = innerLimit;
        }
        const s32 rightLimit = std::min(xend + 1, 256);
        if (fillRight && x < rightLimit) {
            if (first < 0) first = x;
            last = rightLimit;
        }
        first = std::max(first, 0);

        if (last > first) {
            const s32 rl = interpStart->Interpolate(leftCurrent->FinalColor[0], leftNext->FinalColor[0]);
            const s32 gl = interpStart->Interpolate(leftCurrent->FinalColor[1], leftNext->FinalColor[1]);
            const s32 bl = interpStart->Interpolate(leftCurrent->FinalColor[2], leftNext->FinalColor[2]);
            const s32 sl = interpStart->Interpolate(leftCurrent->TexCoords[0], leftNext->TexCoords[0]);
            const s32 tl = interpStart->Interpolate(leftCurrent->TexCoords[1], leftNext->TexCoords[1]);
            const s32 rr = interpEnd->Interpolate(rightCurrent->FinalColor[0], rightNext->FinalColor[0]);
            const s32 gr = interpEnd->Interpolate(rightCurrent->FinalColor[1], rightNext->FinalColor[1]);
            const s32 br = interpEnd->Interpolate(rightCurrent->FinalColor[2], rightNext->FinalColor[2]);
            const s32 sr = interpEnd->Interpolate(rightCurrent->TexCoords[0], rightNext->TexCoords[0]);
            const s32 tr = interpEnd->Interpolate(rightCurrent->TexCoords[1], rightNext->TexCoords[1]);

            // Attributes at both ends of the visible part of the span, at the DS's sample
            // positions (with its own rounding); the GPU interpolates between them,
            // perspective-correctly with W.
            Soft::Interpolator<0> interpX(xstart, xend + 1, wl, wr);
            // Z-buffered depth is computed per pixel in the fragment shader with the DS's own
            // integer interpolation across the span, from the span's edge values, so depth
            // tests come out exactly as on the DS. (Interpolating linearly on the GPU instead
            // flipped them where two surfaces are almost at the same depth, as the haze and
            // the ground at the horizon of Pokémon's battle backgrounds: a dashed line.)
            const u32 spanX = wbuffer ? 0u
                    : static_cast<u32>(xstart) | (static_cast<u32>(xend + 1 - xstart) << 16);
            for (s32 end : {first, last}) {
                interpX.SetX(end);
                s32 z = interpX.InterpolateZ(zl, zr, wbuffer);
                s32 w = wl;
                if (wl != wr && end != xstart) {
                    // W itself is perspective-correct: 1/W is linear across the span.
                    const float f = static_cast<float>(end - xstart)
                            / static_cast<float>(xend + 1 - xstart);
                    w = static_cast<s32>(static_cast<float>(wl) * static_cast<float>(wr)
                            / ((1.0f - f) * static_cast<float>(wr) + f * static_cast<float>(wl)) + 0.5f);
                }
                const s32 r = interpX.Interpolate(rl, rr);
                const s32 g = interpX.Interpolate(gl, gr);
                const s32 b = interpX.Interpolate(bl, br);
                const s32 s = interpX.Interpolate(sl, sr);
                const s32 t = interpX.Interpolate(tl, tr);
                z = std::clamp(z, 0, 0xffffff);
                const u32 packedW = static_cast<u32>(std::clamp(w, 1, 0xffff));
                const u32 color = static_cast<u32>(r >> 1) | (static_cast<u32>(g >> 1) << 8)
                        | (static_cast<u32>(b >> 1) << 16) | (alpha << 24);
                const u32 texcoord = static_cast<u16>(s) | (static_cast<u32>(static_cast<u16>(t)) << 16);
                const u32 px = static_cast<u32>(end) * 16 + 8;
                for (s32 row : {y, y + 1}) {
                    VertexOut[VertexCount++] = {px | (static_cast<u32>(row * 16) << 16),
                            static_cast<u32>(z), packedW, color, texcoord,
                            attr, polygon->TexParam, polygon->TexPalette,
                            static_cast<u32>(zl), static_cast<u32>(zr), spanX};
                }
            }
        }

        rp.XL = rp.SlopeL.Step();
        rp.XR = rp.SlopeR.Step();
    }
}

void VulkanRenderer::AddPolygonTriangles(Polygon* polygon, bool line) {
    const bool splitAroundCenter = BetterPolygons && !line && polygon->NumVertices > 3;
    std::array<u32, 10> vertexIndices{};
    u32 centerIndex = 0;
    if (splitAroundCenter) {
        u32 centerX = 0;
        u32 centerY = 0;
        float centerZ = 0.0f;
        float reciprocalW = 0.0f;
        float centerR = 0.0f;
        float centerG = 0.0f;
        float centerB = 0.0f;
        float centerS = 0.0f;
        float centerT = 0.0f;

        for (u32 vertex = 0; vertex < polygon->NumVertices; ++vertex) {
            const Vertex* source = polygon->Vertices[vertex];
            centerX += source->HiresPosition[0];
            centerY += source->HiresPosition[1];
            const float weightedW = static_cast<float>(polygon->FinalW[vertex])
                    * polygon->NumVertices;
            reciprocalW += 1.0f / weightedW;
            centerZ += polygon->WBuffer
                    ? static_cast<float>(polygon->FinalZ[vertex]) / weightedW
                    : static_cast<float>(polygon->FinalZ[vertex]);
            centerR += static_cast<float>(source->FinalColor[0] >> 1) / weightedW;
            centerG += static_cast<float>(source->FinalColor[1] >> 1) / weightedW;
            centerB += static_cast<float>(source->FinalColor[2] >> 1) / weightedW;
            centerS += static_cast<float>(source->TexCoords[0]) / weightedW;
            centerT += static_cast<float>(source->TexCoords[1]) / weightedW;
        }

        centerX = ((centerX / polygon->NumVertices) * Scale) >> 4;
        centerY = ((centerY / polygon->NumVertices) * Scale) >> 4;
        const float centerW = 1.0f / reciprocalW;
        centerZ = polygon->WBuffer ? centerZ * centerW
                                   : centerZ / polygon->NumVertices;
        centerR *= centerW;
        centerG *= centerW;
        centerB *= centerW;
        centerS *= centerW;
        centerT *= centerW;

        const u32 z = static_cast<u32>(centerZ);
        const u32 attr = PolygonAttr(polygon);
        const u32 alpha = (polygon->Attr >> 16) & 31;
        VertexOut[VertexCount++] = {(centerX << 4) | (centerY << 20), z, static_cast<u32>(centerW),
                static_cast<u32>(centerR)
                        | (static_cast<u32>(centerG) << 8)
                        | (static_cast<u32>(centerB) << 16) | (alpha << 24),
                static_cast<u16>(static_cast<s32>(centerS))
                        | (static_cast<u32>(static_cast<u16>(static_cast<s32>(centerT)))
                           << 16),
                attr, polygon->TexParam, polygon->TexPalette, 0, 0, 0};
        centerIndex = VertexCount - 1;
    }
    for (u32 vertex = 0; vertex < polygon->NumVertices; ++vertex)
        vertexIndices[vertex] = AddVertex(polygon, vertex);
    if (line) {
        if (polygon->NumVertices >= 2) {
            IndexOut[IndexCount++] = vertexIndices[0];
            IndexOut[IndexCount++] = vertexIndices[1];
        }
    } else if (splitAroundCenter) {
        for (u32 vertex = 1; vertex < polygon->NumVertices; ++vertex) {
            IndexOut[IndexCount++] = centerIndex;
            IndexOut[IndexCount++] = vertexIndices[vertex - 1];
            IndexOut[IndexCount++] = vertexIndices[vertex];
        }
        IndexOut[IndexCount++] = centerIndex;
        IndexOut[IndexCount++] = vertexIndices[polygon->NumVertices - 1];
        IndexOut[IndexCount++] = vertexIndices[0];
    } else {
        for (u32 vertex = 2; vertex < polygon->NumVertices; ++vertex) {
            IndexOut[IndexCount++] = vertexIndices[0];
            IndexOut[IndexCount++] = vertexIndices[vertex - 1];
            IndexOut[IndexCount++] = vertexIndices[vertex];
        }
    }
}

const VulkanRenderer::SpanCacheEntry* VulkanRenderer::FindSpans(
        const SpanCache& cache, u64 hash, const u32* key, u32 keySize) const {
    const size_t mask = cache.Slots.size() - 1;
    for (size_t slot = hash & mask; cache.Slots[slot].Generation == cache.Generation;
            slot = (slot + 1) & mask) {
        const SpanCacheEntry& entry = cache.Entries[cache.Slots[slot].Entry];
        if (entry.Hash == hash && entry.KeySize == keySize
                && std::memcmp(cache.Keys.data() + entry.KeyOffset, key,
                               keySize * sizeof(u32)) == 0)
            return &entry;
    }
    return nullptr;
}

void VulkanRenderer::AddSpanCacheEntry(SpanCache& cache, const SpanCacheEntry& entry) {
    // A polygon drawn twice in a frame is entered twice; lookups find the first entry, and
    // both hold the same spans.
    const size_t mask = cache.Slots.size() - 1;
    size_t slot = entry.Hash & mask;
    while (cache.Slots[slot].Generation == cache.Generation) slot = (slot + 1) & mask;
    cache.Slots[slot] = {cache.Generation, static_cast<u32>(cache.Entries.size())};
    cache.Entries.push_back(entry);
}

// At 1x. Stepping a polygon's edges costs hundreds of instructions per scanline on the
// Shield's CPU, and games resubmit most polygons unchanged from one frame to the next (96%
// of them in Pokémon White 2's town and house scenes), so each polygon's spans are kept with
// everything they are computed from, and a polygon that comes back with the same inputs gets
// a copy of the same vertices instead of being stepped again. The key holds every input of
// AddPolygonSpans, and a match compares the whole key (the hash only finds it), so the
// vertices are exactly the ones stepping would produce.
void VulkanRenderer::BuildPolygonSpans() {
    SpanCache& previous = SpanCaches[SpanCacheCurrent ^ 1];
    SpanCache& current = SpanCaches[SpanCacheCurrent];
    current.Keys.clear();
    current.Entries.clear();
    if (++current.Generation == 0) {
        for (auto& slot : current.Slots) slot.Generation = 0;
        current.Generation = 1;
    }
    const VertexGPU* previousVertices = SpanVertices[previous.Vertices].get();
    const u32 written = previous.Vertices ^ 1;
    VertexOut = SpanVertices[written].get();
    // While every polygon so far is one the previous frame had at the same place, the
    // vertices so far are the previous frame's, which are already in place in its buffer (and
    // in the mapped one); nothing is copied until a polygon differs.
    bool unchanged = true;
    const u32 fillAllEdges = (RenderDispCnt & ((1 << 4) | (1 << 5))) != 0 ? 1u : 0u;

    for (u32 i = 0; i < RenderNumPolygons; ++i) {
        Polygon* polygon = RenderPolygonRAM[i];
        if (!polygon || polygon->Degenerate || polygon->IsShadowMask) continue;
        // Wireframe polygons (alpha 0) only draw their edges on the DS; they are left out.
        if (((polygon->Attr >> 16) & 31) == 0) continue;
        const s32 ytop = polygon->YTop;
        const s32 ybottom = polygon->YBottom == ytop ? ytop + 1 : polygon->YBottom;
        const size_t rows = static_cast<size_t>(std::max(ybottom - ytop, 0));
        if (rows == 0 || VertexCount + rows * 4 > MaxVertices
                || IndexCount + rows * 6 > MaxIndices) continue;

        const u32 count = polygon->NumVertices;
        const u32 keySize = 6 + 8 * count;
        const u32 keyOffset = static_cast<u32>(current.Keys.size());
        current.Keys.resize(keyOffset + keySize);
        u32* key = current.Keys.data() + keyOffset;
        key[0] = count | (polygon->VTop << 8) | (polygon->VBottom << 16)
                | (polygon->FacingView ? 1u << 24 : 0u) | (polygon->WBuffer ? 1u << 25 : 0u)
                | (fillAllEdges << 26);
        key[1] = polygon->Attr;
        key[2] = polygon->TexParam;
        key[3] = polygon->TexPalette;
        key[4] = static_cast<u32>(polygon->YTop);
        key[5] = static_cast<u32>(polygon->YBottom);
        u32* vertexKey = key + 6;
        for (u32 v = 0; v < count; ++v, vertexKey += 8) {
            const Vertex* vertex = polygon->Vertices[v];
            vertexKey[0] = static_cast<u32>(vertex->FinalPosition[0]);
            vertexKey[1] = static_cast<u32>(vertex->FinalPosition[1]);
            vertexKey[2] = static_cast<u32>(vertex->FinalColor[0]);
            vertexKey[3] = static_cast<u32>(vertex->FinalColor[1]);
            vertexKey[4] = static_cast<u32>(vertex->FinalColor[2]);
            vertexKey[5] = static_cast<u16>(vertex->TexCoords[0])
                    | (static_cast<u32>(static_cast<u16>(vertex->TexCoords[1])) << 16);
            vertexKey[6] = static_cast<u32>(polygon->FinalZ[v]);
            vertexKey[7] = static_cast<u32>(polygon->FinalW[v]);
        }
        const u64 hash = XXH3_64bits(key, keySize * sizeof(u32));

        const u32 first = VertexCount;
        const SpanCacheEntry* cached = FindSpans(previous, hash, key, keySize);
        if (unchanged && !(cached && cached->FirstVertex == first)) {
            unchanged = false;
            std::memcpy(VertexOut, previousVertices, first * sizeof(VertexGPU));
        }
        if (!cached) {
            AddPolygonSpans(polygon);
        } else {
            if (!unchanged)
                std::memcpy(VertexOut + first, previousVertices + cached->FirstVertex,
                            cached->VertexCount * sizeof(VertexGPU));
            VertexCount += cached->VertexCount;
        }
        AddSpanCacheEntry(current, {hash, keyOffset, keySize, first, VertexCount - first});

        const u32 indices = (VertexCount - first) / 4 * 6;
        IndexCount += indices;
        if (indices) {
            DrawCall draw{};
            draw.FirstIndex = 0;
            draw.IndexCount = indices;
            draw.VertexOffset = static_cast<s32>(first);
            draw.Translucent = polygon->Translucent;
            draw.DepthEqual = (polygon->Attr & (1 << 14)) != 0;
            draw.DepthWrite = !polygon->Translucent || (polygon->Attr & (1 << 11));
            draw.Line = false;
            draw.WBuffer = polygon->WBuffer;
            draw.PolygonID = static_cast<u8>((polygon->Attr >> 24) & 0x3f);
            // Texture formats can make an otherwise opaque polygon partially translucent.
            // The DS first writes its fully opaque texels to depth, then blends the rest.
            if (draw.Translucent && ((polygon->Attr >> 16) & 31) == 31) {
                DrawCall opaque = draw;
                opaque.Translucent = false;
                opaque.DepthWrite = true;
                Draws.push_back(opaque);
            }
            Draws.push_back(draw);
        }
        if (IndexCount >= MaxIndices) break;
    }

    current.Vertices = unchanged ? previous.Vertices : written;
    // One sequential copy into the mapped (write-combined) buffer, and none at all when the
    // frame's vertices are the ones it already holds.
    if (VertexCount && !(unchanged && SpanVerticesUploaded)) {
        std::memcpy(VertexBuffer.Mapped, SpanVertices[current.Vertices].get(),
                    VertexCount * sizeof(VertexGPU));
    }
    SpanVerticesUploaded = true;
    SpanCacheCurrent ^= 1;
}

void VulkanRenderer::BuildPolygons() {
    VertexCount = 0;
    IndexCount = 0;
    Draws.clear();
    if (Scale == 1) {
        BuildPolygonSpans();
        return;
    }
    VertexOut = static_cast<VertexGPU*>(VertexBuffer.Mapped);
    IndexOut = static_cast<u32*>(IndexBuffer.Mapped);
    // The triangles overwrite the span vertices the mapped buffer held.
    SpanVerticesUploaded = false;
    for (u32 i = 0; i < RenderNumPolygons; ++i) {
        Polygon* polygon = RenderPolygonRAM[i];
        if (!polygon || polygon->Degenerate || polygon->IsShadowMask) continue;
        // The triangle path needs one vertex more than the polygon has when it is split
        // around its center.
        if (VertexCount + polygon->NumVertices + 1 > MaxVertices) continue;
        DrawCall draw{};
        draw.FirstIndex = static_cast<u32>(IndexCount);
        draw.Translucent = polygon->Translucent;
        draw.DepthEqual = (polygon->Attr & (1 << 14)) != 0;
        draw.DepthWrite = !polygon->Translucent || (polygon->Attr & (1 << 11));
        draw.Line = polygon->Type == 1;
        draw.WBuffer = polygon->WBuffer;
        draw.PolygonID = static_cast<u8>((polygon->Attr >> 24) & 0x3f);
        AddPolygonTriangles(polygon, draw.Line);
        draw.IndexCount = static_cast<u32>(IndexCount) - draw.FirstIndex;
        if (draw.IndexCount) {
            // Texture formats can make an otherwise opaque polygon partially translucent.
            // The DS first writes its fully opaque texels to depth, then blends the rest.
            if (draw.Translucent && ((polygon->Attr >> 16) & 31) == 31) {
                DrawCall opaque = draw;
                opaque.Translucent = false;
                opaque.DepthWrite = true;
                Draws.push_back(opaque);
            }
            Draws.push_back(draw);
        }
        if (IndexCount >= MaxIndices) break;
    }
}

VkPipeline VulkanRenderer::PipelineFor(const DrawCall& draw) const {
    const u32 key = (draw.Translucent ? 1u : 0u) | (draw.DepthEqual ? 2u : 0u)
            | (draw.DepthWrite ? 4u : 0u) | (draw.Line ? 8u : 0u);
    return Pipelines[key];
}

bool VulkanRenderer::CompleteSubmission() {
    if (!SubmissionPending) return true;
    if (vkWaitForFences(Device, 1, &Fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        Ready = false;
        SubmissionPending = false;
        return Fail("frame completion failed");
    }
    SubmissionPending = false;
    Buffer& completed = ReadbackBuffers[ReadbackWriteIndex];
    Invalidate(completed);
    const auto* source = static_cast<const u32*>(completed.Mapped);
    if (Scale == 1) {
        // GPU2D consumes this persistently mapped slot while the next GPU3D frame writes the
        // other slot. This removes the old 192 KiB readback-to-RawOutput copy every 3D frame.
        ReadbackOutput = source;
    } else {
        for (u32 y = 0; y < 192; ++y)
            for (u32 x = 0; x < 256; ++x)
                RawOutput[y * 256 + x] = source[(y * Scale) * Width + x * Scale];
        ReadbackOutput = RawOutput.data();
    }
    ReadbackWriteIndex ^= 1u;
    return true;
}

void VulkanRenderer::FillEmptyOutput(u32 clear) {
    const auto expand5 = [](u32 value) { return (value << 3) | (value >> 2); };
    const u32 red = expand5(clear & 31);
    const u32 green = expand5((clear >> 5) & 31);
    const u32 blue = expand5((clear >> 10) & 31);
    const u32 alpha = expand5((clear >> 16) & 31);
    RawOutput.fill(red | (green << 8) | (blue << 16) | (alpha << 24));
    ReadbackOutput = RawOutput.data();
}

// Whether any polygon gets as far as being drawn (BuildPolygons may still find that the ones
// that do cover no pixel).
bool VulkanRenderer::HasDrawablePolygons() const {
    for (u32 i = 0; i < RenderNumPolygons; ++i) {
        const Polygon* polygon = RenderPolygonRAM[i];
        if (!polygon || polygon->Degenerate || polygon->IsShadowMask) continue;
        // At 1x, as BuildPolygonSpans: no wireframe polygons, and none without a row.
        if (Scale == 1 && (((polygon->Attr >> 16) & 31) == 0 || polygon->YBottom < polygon->YTop))
            continue;
        return true;
    }
    return false;
}

// The polygons are turned into spans or triangles and the frame is recorded and submitted on
// a thread of its own, while the emulation goes on. It reads only what the next VBlank
// replaces (the polygon and vertex RAM the frame was given and the render registers), and
// VCount144, which melonDS calls just before that VBlank, waits for it, as it waits for the
// software renderer's thread. Its result is read back one frame later, as before, so frames
// come out exactly as they did when all of this ran here.
void VulkanRenderer::WorkerMain() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "melonDS 3D");
#endif
    std::unique_lock<std::mutex> lock(WorkerLock);
    for (;;) {
        WorkerWake.wait(lock, [this] { return WorkerBusy || WorkerQuit; });
        if (WorkerQuit) return;
        lock.unlock();
        BuildAndSubmit();
        lock.lock();
        WorkerBusy = false;
        WorkerIdle.notify_all();
    }
}

void VulkanRenderer::FinishRendering() {
    if (!Worker.joinable()) return;
    std::unique_lock<std::mutex> lock(WorkerLock);
    WorkerIdle.wait(lock, [this] { return !WorkerBusy; });
}

void VulkanRenderer::StopWorker() {
    if (!Worker.joinable()) return;
    FinishRendering();
    {
        std::lock_guard<std::mutex> lock(WorkerLock);
        WorkerQuit = true;
    }
    WorkerWake.notify_one();
    Worker.join();
    WorkerQuit = false;
}

void VulkanRenderer::RenderFrame() {
    FinishRendering();
    if (!Ready) return;
    // The previous frame's GPU work has normally finished long ago (a whole DS frame of
    // emulation ran since its submit), so waiting first costs nothing and lets the polygons
    // be written straight into the mapped vertex buffer.
    const auto waitStart = TraceClock::now();
    if (!CompleteSubmission()) return;
    // A frame whose polygons all turned out to cover nothing: its picture is the clear color,
    // shown when a drawn frame's would be.
    if (EmptyFramePending) {
        EmptyFramePending = false;
        FillEmptyOutput(EmptyFrameClear);
    }
    const auto waitEnd = TraceClock::now();
    FrameWaitNanoseconds = ElapsedNanoseconds(waitStart, waitEnd);

    const bool vramChanged = UploadMemory();
    FrameUploadNanoseconds = ElapsedNanoseconds(waitEnd, TraceClock::now());
    // Like the software renderer: when no new geometry was flushed and none of the 3D
    // registers or textures changed, the last output is still right.
    if (RenderFrameIdentical && !vramChanged && OutputCurrent) return;
    OutputCurrent = true;

    // Most menus are entirely 2D. Avoid a Vulkan submit and readback when the 3D engine has
    // nothing to draw.
    if (!HasDrawablePolygons()) {
        FillEmptyOutput(RenderClearAttr1);
        RecordTrace(0, FrameWaitNanoseconds, FrameUploadNanoseconds, 0, RenderNumPolygons, 0, 0,
                    true);
        return;
    }

    if (!Worker.joinable()) {
        BuildAndSubmit();
        return;
    }
    {
        std::lock_guard<std::mutex> lock(WorkerLock);
        WorkerBusy = true;
    }
    WorkerWake.notify_one();
}

void VulkanRenderer::BuildAndSubmit() {
    const auto buildStart = TraceClock::now();
    BuildPolygons();
    const auto buildEnd = TraceClock::now();
    const u64 buildNanoseconds = ElapsedNanoseconds(buildStart, buildEnd);
    const u64 waitNanoseconds = FrameWaitNanoseconds;

    if (Draws.empty()) {
        EmptyFramePending = true;
        EmptyFrameClear = RenderClearAttr1;
        RecordTrace(buildNanoseconds, waitNanoseconds, FrameUploadNanoseconds, 0,
                    RenderNumPolygons, Draws.size(), VertexCount, true);
        return;
    }

    auto* config = static_cast<ShaderConfig*>(UniformBuffer.Mapped);
    std::memset(config, 0, sizeof(*config));
    config->ScreenSize[0] = static_cast<float>(Width);
    config->ScreenSize[1] = static_cast<float>(Height);
    config->DispCnt = RenderDispCnt;
    for (int i = 0; i < 32; ++i) {
        const u16 color = RenderToonTable[i];
        config->ToonColors[i][0] = static_cast<float>(color & 31) / 31.0f;
        config->ToonColors[i][1] = static_cast<float>((color >> 5) & 31) / 31.0f;
        config->ToonColors[i][2] = static_cast<float>((color >> 10) & 31) / 31.0f;
    }
    Flush(UniformBuffer);
    Flush(VertexBuffer);
    Flush(IndexBuffer);
    const auto uploadEnd = TraceClock::now();

    vkResetFences(Device, 1, &Fence);
    vkResetCommandBuffer(CommandBuffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(CommandBuffer, &begin);

    const u32 clear = RenderClearAttr1;
    VkClearValue clears[2]{};
    clears[0].color.float32[0] = static_cast<float>((clear >> 10) & 31) / 31.0f;
    clears[0].color.float32[1] = static_cast<float>((clear >> 5) & 31) / 31.0f;
    clears[0].color.float32[2] = static_cast<float>(clear & 31) / 31.0f;
    clears[0].color.float32[3] = static_cast<float>((clear >> 16) & 31) / 31.0f;
    const u32 clearZ = ((RenderClearAttr2 & 0x7fff) * 0x200) + 0x1ff;
    clears[1].depthStencil = {static_cast<float>(clearZ) / 16777216.0f, 0};
    VkRenderPassBeginInfo renderBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderBegin.renderPass = RenderPass;
    renderBegin.framebuffer = FramebufferObject;
    renderBegin.renderArea.extent = {Width, Height};
    renderBegin.clearValueCount = 2;
    renderBegin.pClearValues = clears;
    vkCmdBeginRenderPass(CommandBuffer, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
    VkRect2D scissor{{0, 0}, {Width, Height}};
    vkCmdSetViewport(CommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(CommandBuffer, 0, 1, &scissor);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(CommandBuffer, 0, 1, &VertexBuffer.Handle, &offset);
    if (Scale == 1)
        vkCmdBindIndexBuffer(CommandBuffer, SpanIndexBuffer.Handle, 0, VK_INDEX_TYPE_UINT16);
    else
        vkCmdBindIndexBuffer(CommandBuffer, IndexBuffer.Handle, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout,
                            0, 1, &DescriptorSet, 0, nullptr);
    VkPipeline current = VK_NULL_HANDLE;
    u32 currentStencil = ~0u;
    for (const auto& draw : Draws) {
        VkPipeline pipeline = PipelineFor(draw);
        if (pipeline != current) {
            vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            current = pipeline;
        }
        const u32 stencil = draw.Translucent ? 0x40u | draw.PolygonID : 0u;
        if (stencil != currentStencil) {
            vkCmdSetStencilReference(CommandBuffer, VK_STENCIL_FACE_FRONT_AND_BACK, stencil);
            currentStencil = stencil;
        }
        PushConstants push{draw.WBuffer ? 1u : 0u, draw.Translucent ? 1u : 0u};
        vkCmdPushConstants(CommandBuffer, PipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(push), &push);
        vkCmdDrawIndexed(CommandBuffer, draw.IndexCount, 1, draw.FirstIndex, draw.VertexOffset,
                         0);
    }
    vkCmdEndRenderPass(CommandBuffer);
    VkImageMemoryBarrier readbackBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    readbackBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    readbackBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    readbackBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.image = ColorImage;
    readbackBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    readbackBarrier.subresourceRange.levelCount = 1;
    readbackBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &readbackBarrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {Width, Height, 1};
    vkCmdCopyImageToBuffer(CommandBuffer, ColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           ReadbackBuffers[ReadbackWriteIndex].Handle, 1, &copy);
    // Makes the copy visible to the CPU's reads once the fence signals.
    VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostBarrier.buffer = ReadbackBuffers[ReadbackWriteIndex].Handle;
    hostBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &hostBarrier, 0, nullptr);
    vkEndCommandBuffer(CommandBuffer);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &CommandBuffer;
    if (vkQueueSubmit(Queue, 1, &submit, Fence) != VK_SUCCESS) {
        Ready = false;
        Fail("frame submission failed");
        return;
    }
    SubmissionPending = true;
    const auto submitEnd = TraceClock::now();
    RecordTrace(buildNanoseconds, waitNanoseconds,
                FrameUploadNanoseconds + ElapsedNanoseconds(buildEnd, uploadEnd),
                ElapsedNanoseconds(uploadEnd, submitEnd), RenderNumPolygons,
                Draws.size(), VertexCount, false);
}

u32* VulkanRenderer::GetLine(int line) {
    const u32* source = &ReadbackOutput[static_cast<size_t>(line) * 256];
    u32* destination = &Output[static_cast<size_t>(line) * 256];
#if defined(__ARM_NEON)
    const uint32x4_t rgbMask = vdupq_n_u32(0x00fcfcfc);
    const uint32x4_t alphaMask = vdupq_n_u32(0xf8000000);
    for (int x = 0; x < 256; x += 4) {
        const uint32x4_t pixels = vld1q_u32(source + x);
        const uint32x4_t rgb = vshrq_n_u32(vandq_u32(pixels, rgbMask), 2);
        const uint32x4_t alpha = vshrq_n_u32(vandq_u32(pixels, alphaMask), 3);
        vst1q_u32(destination + x, vorrq_u32(rgb, alpha));
    }
#else
    const u64* source64 = reinterpret_cast<const u64*>(source);
    u64* destination64 = reinterpret_cast<u64*>(destination);
    for (int x = 0; x < 256; x += 2) {
        const u64 pixels = *source64++;
        const u64 rgb = pixels & 0x00fcfcfc00fcfcfcULL;
        const u64 alpha = pixels & 0xf8000000f8000000ULL;
        *destination64++ = (rgb >> 2) | (alpha >> 3);
    }
#endif
    return &Output[static_cast<size_t>(line) * 256];
}

extern "C" __attribute__((visibility("default")))
bool melonds_vulkan_get_shared_context(MelonDSVulkanSharedContext* context) {
    if (context == nullptr || GPU::Renderer != 2 || !GPU3D::CurrentRenderer) return false;
    auto* renderer = dynamic_cast<VulkanRenderer*>(GPU3D::CurrentRenderer.get());
    return renderer != nullptr && renderer->ExportSharedContext(*context);
}

} // namespace GPU3D
