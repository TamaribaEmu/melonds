/* Experimental Vulkan renderer for the Android libretro port. */
#pragma once

#if defined(__ANDROID__) && !defined(VK_USE_PLATFORM_ANDROID_KHR)
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif

#include "GPU3D.h"
#include "melonds_vulkan_shared.h"

#include <vulkan/vulkan.h>

#include <array>
#include <string>
#include <vector>

namespace GPU3D {

class VulkanRenderer final : public Renderer3D {
public:
    VulkanRenderer();
    ~VulkanRenderer() override;

    bool Init() override;
    void DeInit() override;
    void Reset() override;
    void SetRenderSettings(GPU::RenderSettings& settings) override;
    void RenderFrame() override;
    u32* GetLine(int line) override;
    bool ExportSharedContext(MelonDSVulkanSharedContext& context) const;

private:
    struct VertexGPU {
        u32 XY;        // 12.4 fixed point
        u32 Z;         // 24-bit Z, or W when W-buffering
        u32 W;         // the polygon's normalized 16-bit W
        u32 Color;
        u32 Texcoord;
        u32 Attr;
        u32 TexParam;
        u32 TexPalette;
        // At 1x, the span's Z at both edges and its start | width << 16 (width 0: the
        // hardware's depth), for the DS's per-pixel Z interpolation in the fragment shader.
        u32 SpanZ0;
        u32 SpanZ1;
        u32 SpanX;
    };

    struct DrawCall {
        u32 FirstIndex;
        u32 IndexCount;
        bool Translucent;
        bool DepthEqual;
        bool DepthWrite;
        bool Line;
        bool WBuffer;
        u8 PolygonID;
    };

    struct ShaderConfig {
        float ScreenSize[2];
        u32 DispCnt;
        u32 Padding;
        float ToonColors[32][4];
    };

    struct Buffer {
        VkBuffer Handle = VK_NULL_HANDLE;
        VkDeviceMemory Memory = VK_NULL_HANDLE;
        void* Mapped = nullptr;
        VkDeviceSize Size = 0;
        bool Coherent = false;
    };

    bool CreateDevice();
    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result,
                      bool readback = false);
    bool CreateTargets();
    bool CreateDescriptors();
    bool CreatePipelineCache();
    bool CreatePipelines();
    bool CreatePipeline(bool translucent, bool depthEqual, bool depthWrite, bool line,
                        VkPipeline& result);
    bool CreateShader(const u8* bytes, size_t size, VkShaderModule& result);
    u32 FindMemoryType(u32 bits, VkMemoryPropertyFlags required,
                       VkMemoryPropertyFlags preferred = 0) const;
    void DestroyBuffer(Buffer& buffer);
    void DestroyTargets();
    bool UploadMemory();
    void BuildPolygons();
    bool AddPolygonSpans(Polygon* polygon);
    void AddPolygonTriangles(Polygon* polygon, bool line);
    bool CompleteSubmission();
    u32 AddVertex(Polygon* polygon, u32 index);
    VkPipeline PipelineFor(const DrawCall& draw) const;
    void Flush(Buffer& buffer);
    void Invalidate(Buffer& buffer);
    void SavePipelineCache();
    bool Fail(const char* message);

    VkInstance Instance = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties MemoryProperties{};
    VkDevice Device = VK_NULL_HANDLE;
    VkQueue Queue = VK_NULL_HANDLE;
    u32 QueueFamily = ~0u;
    VkCommandPool CommandPool = VK_NULL_HANDLE;
    VkCommandBuffer CommandBuffer = VK_NULL_HANDLE;
    VkFence Fence = VK_NULL_HANDLE;

    VkImage ColorImage = VK_NULL_HANDLE;
    VkDeviceMemory ColorMemory = VK_NULL_HANDLE;
    VkImage DepthImage = VK_NULL_HANDLE;
    VkDeviceMemory DepthMemory = VK_NULL_HANDLE;
    VkFormat DepthFormat = VK_FORMAT_UNDEFINED;
    VkImageView ColorView = VK_NULL_HANDLE;
    VkImageView DepthView = VK_NULL_HANDLE;
    VkRenderPass RenderPass = VK_NULL_HANDLE;
    VkFramebuffer FramebufferObject = VK_NULL_HANDLE;

    VkDescriptorSetLayout DescriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet DescriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout PipelineLayout = VK_NULL_HANDLE;
    VkPipelineCache PipelineCache = VK_NULL_HANDLE;
    std::string PipelineCachePath;
    std::vector<u8> PipelineCacheDiskData;
    VkShaderModule VertexShader = VK_NULL_HANDLE;
    VkShaderModule FragmentShader = VK_NULL_HANDLE;
    std::array<VkPipeline, 16> Pipelines{};

    Buffer UniformBuffer;
    Buffer TextureBuffer;
    Buffer PaletteBuffer;
    Buffer VertexBuffer;
    Buffer IndexBuffer;
    std::array<Buffer, 2> ReadbackBuffers{};

    // The frame's geometry, written straight into the mapped vertex and index buffers.
    VertexGPU* VertexOut = nullptr;
    u32* IndexOut = nullptr;
    u32 VertexCount = 0;
    u32 IndexCount = 0;
    std::vector<DrawCall> Draws;
    std::array<u32, 256 * 192> RawOutput{};
    std::array<u32, 256 * 192> Output{};
    const u32* ReadbackOutput = RawOutput.data();
    u32 ReadbackWriteIndex = 0;
    int Scale = 1;
    bool BetterPolygons = false;
    u32 Width = 256;
    u32 Height = 192;
    bool Ready = false;
    bool SubmissionPending = false;
    bool OutputCurrent = false;
};

} // namespace GPU3D
