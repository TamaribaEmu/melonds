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

#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <unistd.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace GPU3D {

namespace {
constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr size_t MaxVertices = 10240;
constexpr size_t MaxIndices = 2048 * 40;

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
    const char* instanceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    };
    instanceInfo.enabledExtensionCount = 2;
    instanceInfo.ppEnabledExtensionNames = instanceExtensions;
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
    const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;
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

bool VulkanRenderer::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result) {
    result.Size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(Device, &info, nullptr, &result.Handle) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(Device, result.Handle, &requirements);
    const u32 type = FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
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
    for (VkFormat candidate : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
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
    return CreateBuffer(readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, ReadbackBuffers[0])
            && CreateBuffer(readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, ReadbackBuffers[1]);
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
        {0, 0, VK_FORMAT_R16G16B16A16_UINT, 0},
        {1, 0, VK_FORMAT_R8G8B8A8_UINT, 8},
        {2, 0, VK_FORMAT_R16G16_SINT, 12},
        {3, 0, VK_FORMAT_R32G32B32_UINT, 16},
    };
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 4;
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
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
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
            || !CreateBuffer(MaxIndices * sizeof(u16), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                             IndexBuffer)) return Fail("could not allocate upload buffers");
    if (!CreateTargets() || !CreateDescriptors() || !CreatePipelines()) return false;
    Vertices.reserve(MaxVertices);
    Indices.reserve(MaxIndices);
    Draws.reserve(2048);
    Ready = true;
    __android_log_print(ANDROID_LOG_INFO, "MelonDSVulkan3D",
            "Initialized %ux%u with mapped ping-pong readback", Width, Height);
    return true;
}

void VulkanRenderer::DeInit() {
    if (!Device) return;
    vkDeviceWaitIdle(Device);
    SavePipelineCache();
    DestroyTargets();
    DestroyBuffer(UniformBuffer);
    DestroyBuffer(TextureBuffer);
    DestroyBuffer(PaletteBuffer);
    DestroyBuffer(VertexBuffer);
    DestroyBuffer(IndexBuffer);
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
    if (Device && SubmissionPending) {
        vkWaitForFences(Device, 1, &Fence, VK_TRUE, UINT64_MAX);
        SubmissionPending = false;
    }
    RawOutput.fill(0);
    Output.fill(0);
    ReadbackOutput = RawOutput.data();
    ReadbackWriteIndex = 0;
}

void VulkanRenderer::SetRenderSettings(GPU::RenderSettings& settings) {
    BetterPolygons = settings.GL_BetterPolygons;
    const int requested = std::clamp(settings.GL_ScaleFactor, 1, 4);
    if (requested == Scale || !Ready) return;
    vkDeviceWaitIdle(Device);
    SubmissionPending = false;
    DestroyTargets();
    Scale = requested;
    if (!CreateTargets() || !CreatePipelines()) {
        Ready = false;
        Fail("could not resize render targets");
    }
}

void VulkanRenderer::UploadMemory() {
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
}

u32 VulkanRenderer::AddVertex(Polygon* polygon, u32 index) {
    Vertex* vertex = polygon->Vertices[index];
    u32 z = polygon->FinalZ[index];
    const u32 w = polygon->FinalW[index];
    u32 shift = 0;
    while (z > 0xffff) { z >>= 1; ++shift; }
    u32 x = Scale > 1 ? (vertex->HiresPosition[0] * Scale) >> 4 : vertex->FinalPosition[0];
    u32 y = Scale > 1 ? (vertex->HiresPosition[1] * Scale) >> 4 : vertex->FinalPosition[1];
    u32 attr = polygon->Attr & 0x1f00c8f0;
    if (polygon->FacingView) attr |= 1 << 8;
    if (polygon->WBuffer) attr |= 1 << 9;
    attr |= shift << 16;
    const u32 alpha = (polygon->Attr >> 16) & 31;
    Vertices.push_back({x | (y << 16), z | (w << 16),
        static_cast<u32>(vertex->FinalColor[0] >> 1)
                | (static_cast<u32>(vertex->FinalColor[1] >> 1) << 8)
                | (static_cast<u32>(vertex->FinalColor[2] >> 1) << 16) | (alpha << 24),
        static_cast<u16>(vertex->TexCoords[0])
                | (static_cast<u32>(static_cast<u16>(vertex->TexCoords[1])) << 16),
        attr, polygon->TexParam, polygon->TexPalette});
    return static_cast<u32>(Vertices.size() - 1);
}

void VulkanRenderer::BuildPolygons() {
    Vertices.clear();
    Indices.clear();
    Draws.clear();
    for (u32 i = 0; i < RenderNumPolygons; ++i) {
        Polygon* polygon = RenderPolygonRAM[i];
        const bool splitAroundCenter = polygon && BetterPolygons && polygon->Type != 1
                && polygon->NumVertices > 3;
        const size_t requiredVertices = polygon
                ? polygon->NumVertices + (splitAroundCenter ? 1u : 0u) : 0u;
        if (!polygon || polygon->Degenerate || polygon->IsShadowMask
                || Vertices.size() + requiredVertices > MaxVertices) continue;
        DrawCall draw{};
        draw.FirstIndex = static_cast<u32>(Indices.size());
        draw.Translucent = polygon->Translucent;
        draw.DepthEqual = (polygon->Attr & (1 << 14)) != 0;
        draw.DepthWrite = !polygon->Translucent || (polygon->Attr & (1 << 11));
        draw.Line = polygon->Type == 1;
        draw.WBuffer = polygon->WBuffer;
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

            u32 z = static_cast<u32>(centerZ);
            u32 shift = 0;
            while (z > 0xffff) { z >>= 1; ++shift; }
            u32 attr = polygon->Attr & 0x1f00c8f0;
            if (polygon->FacingView) attr |= 1 << 8;
            if (polygon->WBuffer) attr |= 1 << 9;
            attr |= shift << 16;
            const u32 alpha = (polygon->Attr >> 16) & 31;
            Vertices.push_back({centerX | (centerY << 16),
                    z | (static_cast<u32>(centerW) << 16),
                    static_cast<u32>(centerR)
                            | (static_cast<u32>(centerG) << 8)
                            | (static_cast<u32>(centerB) << 16) | (alpha << 24),
                    static_cast<u16>(static_cast<s32>(centerS))
                            | (static_cast<u32>(static_cast<u16>(static_cast<s32>(centerT)))
                               << 16),
                    attr, polygon->TexParam, polygon->TexPalette});
            centerIndex = static_cast<u32>(Vertices.size() - 1);
        }
        for (u32 vertex = 0; vertex < polygon->NumVertices; ++vertex)
            vertexIndices[vertex] = AddVertex(polygon, vertex);
        if (draw.Line) {
            if (polygon->NumVertices >= 2) {
                Indices.push_back(static_cast<u16>(vertexIndices[0]));
                Indices.push_back(static_cast<u16>(vertexIndices[1]));
            }
        } else if (splitAroundCenter) {
            for (u32 vertex = 1; vertex < polygon->NumVertices; ++vertex) {
                Indices.push_back(static_cast<u16>(centerIndex));
                Indices.push_back(static_cast<u16>(vertexIndices[vertex - 1]));
                Indices.push_back(static_cast<u16>(vertexIndices[vertex]));
            }
            Indices.push_back(static_cast<u16>(centerIndex));
            Indices.push_back(static_cast<u16>(vertexIndices[polygon->NumVertices - 1]));
            Indices.push_back(static_cast<u16>(vertexIndices[0]));
        } else {
            for (u32 vertex = 2; vertex < polygon->NumVertices; ++vertex) {
                Indices.push_back(static_cast<u16>(vertexIndices[0]));
                Indices.push_back(static_cast<u16>(vertexIndices[vertex - 1]));
                Indices.push_back(static_cast<u16>(vertexIndices[vertex]));
            }
        }
        draw.IndexCount = static_cast<u32>(Indices.size()) - draw.FirstIndex;
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
        if (Indices.size() >= MaxIndices) break;
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

void VulkanRenderer::RenderFrame() {
    if (!Ready) return;
    const auto buildStart = TraceClock::now();
    // Polygon preparation only writes CPU-side vectors. Do it before waiting for the previous
    // GPU submission so the work can overlap when the queue is still finishing its readback.
    BuildPolygons();
    const auto buildEnd = TraceClock::now();
    const auto waitStart = buildEnd;
    if (!CompleteSubmission()) return;
    const auto waitEnd = TraceClock::now();
    const u64 buildNanoseconds = ElapsedNanoseconds(buildStart, buildEnd);
    const u64 waitNanoseconds = ElapsedNanoseconds(waitStart, waitEnd);

    // Most menus are entirely 2D. Avoid a Vulkan submit, full VRAM upload and synchronous
    // readback when the 3D engine has nothing to draw.
    if (Draws.empty()) {
        const auto expand5 = [](u32 value) { return (value << 3) | (value >> 2); };
        const u32 clear = RenderClearAttr1;
        const u32 red = expand5(clear & 31);
        const u32 green = expand5((clear >> 5) & 31);
        const u32 blue = expand5((clear >> 10) & 31);
        const u32 alpha = expand5((clear >> 16) & 31);
        RawOutput.fill(red | (green << 8) | (blue << 16) | (alpha << 24));
        ReadbackOutput = RawOutput.data();
        RecordTrace(buildNanoseconds, waitNanoseconds, 0, 0, RenderNumPolygons,
                    Draws.size(), Vertices.size(), true);
        return;
    }

    const auto uploadStart = TraceClock::now();
    UploadMemory();

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
    std::memcpy(VertexBuffer.Mapped, Vertices.data(), Vertices.size() * sizeof(VertexGPU));
    std::memcpy(IndexBuffer.Mapped, Indices.data(), Indices.size() * sizeof(u16));
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
    clears[1].depthStencil = {static_cast<float>(clearZ) / 16777216.0f, 0xff};
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
    vkCmdBindIndexBuffer(CommandBuffer, IndexBuffer.Handle, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout,
                            0, 1, &DescriptorSet, 0, nullptr);
    VkPipeline current = VK_NULL_HANDLE;
    for (const auto& draw : Draws) {
        VkPipeline pipeline = PipelineFor(draw);
        if (pipeline != current) {
            vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            current = pipeline;
        }
        PushConstants push{draw.WBuffer ? 1u : 0u, draw.Translucent ? 1u : 0u};
        vkCmdPushConstants(CommandBuffer, PipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(push), &push);
        vkCmdDrawIndexed(CommandBuffer, draw.IndexCount, 1, draw.FirstIndex, 0, 0);
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
                ElapsedNanoseconds(uploadStart, uploadEnd),
                ElapsedNanoseconds(uploadEnd, submitEnd), RenderNumPolygons,
                Draws.size(), Vertices.size(), false);
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
