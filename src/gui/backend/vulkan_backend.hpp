/**
 * @file    vulkan_backend.hpp
 * @brief   Vulkan 1.2 Render Backend
 * @license MIT
 *
 * @details
 * Vulkan backend for platforms that enable ENABLE_VULKAN.
 *
 * Design mirrors the OpenGL / D3D11 backends:
 *   - One swapchain owned by the backend
 *   - Minimal command-buffer-per-frame recording pattern
 *   - ImGui integration via imgui_impl_vulkan
 *   - Texture storage managed in an unordered_map, keyed by uint64_t handles
 *
 * All Vulkan-specific types are forward-declared in the header when possible
 * so the header stays lightweight for consumers.
 */

#pragma once

#include "gui/backend/render_backend.hpp"

#if defined(GWT_HAS_VULKAN)

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

struct SDL_Window;

namespace gwt::gui {

class VulkanBackend final : public IRenderBackend {
public:
    VulkanBackend() = default;
    ~VulkanBackend() override;

    // =========================================================================
    // Static availability check
    // =========================================================================
    /**
     * Lightweight availability check. Loads the Vulkan loader via SDL,
     * attempts to create a throwaway VkInstance, destroys it, and returns
     * whether that all succeeded.
     */
    [[nodiscard]] static bool is_available() noexcept;

    // =========================================================================
    // Lifecycle
    // =========================================================================
    [[nodiscard]] bool init(SDL_Window* window) override;
    void shutdown() override;

    // =========================================================================
    // ImGui Integration
    // =========================================================================
    void imgui_init() override;
    void imgui_shutdown() override;
    void imgui_new_frame() override;
    void imgui_render() override;

    // =========================================================================
    // Frame Management
    // =========================================================================
    void begin_frame() override;
    void end_frame() override;
    void present() override;
    void on_resize(int width, int height) override;

    // =========================================================================
    // Texture Operations
    // =========================================================================
    [[nodiscard]] TextureHandle
    create_texture(const TextureDesc& desc, std::span<const uint8_t> data = {}) override;

    void update_texture(TextureHandle handle, std::span<const uint8_t> data) override;
    void destroy_texture(TextureHandle handle) override;

    [[nodiscard]] void* get_imgui_texture_id(TextureHandle handle) const override;

    // =========================================================================
    // Backend Info
    // =========================================================================
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] BackendType type() const noexcept override { return BackendType::Vulkan; }
    [[nodiscard]] bool supports_compute() const noexcept override { return true; }

private:
    // --- Setup helpers ---
    bool create_instance();
    bool pick_physical_device();
    bool create_logical_device();
    bool create_surface();
    bool create_swapchain();
    bool create_image_views();
    bool create_render_pass();
    bool create_framebuffers();
    bool create_command_pool();
    bool create_command_buffers();
    bool create_sync_objects();
    bool create_descriptor_pool();

    void cleanup_swapchain();
    bool recreate_swapchain();

    // --- Texture upload helpers ---
    [[nodiscard]] uint32_t find_memory_type(uint32_t type_filter,
                                            VkMemoryPropertyFlags props) const;
    [[nodiscard]] bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags props,
                                     VkBuffer& buffer, VkDeviceMemory& memory) const;
    [[nodiscard]] VkCommandBuffer begin_single_time() const;
    void end_single_time(VkCommandBuffer cmd) const;
    // Transition UNDEFINED -> upload `pixels` (RGBA8, w*h*4 bytes) -> SHADER_READ.
    void upload_rgba_to_image(VkImage image, const void* pixels,
                              uint32_t w, uint32_t h) const;

    // --- Texture storage (ImGui-registered VkDescriptorSet as ImTextureID) ---
    struct TextureData {
        VkImage        image{VK_NULL_HANDLE};
        VkImageView    view{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        VkSampler      sampler{VK_NULL_HANDLE};
        VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
        TextureDesc    desc;
    };

    // --- SDL / window ---
    SDL_Window* m_window{nullptr};
    int         m_window_width{0};
    int         m_window_height{0};
    bool        m_initialized{false};
    bool        m_swapchain_dirty{false};

    // --- Vulkan core ---
    VkInstance       m_instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT m_debug_messenger{VK_NULL_HANDLE};
    VkSurfaceKHR     m_surface{VK_NULL_HANDLE};
    VkPhysicalDevice m_physical_device{VK_NULL_HANDLE};
    VkDevice         m_device{VK_NULL_HANDLE};
    uint32_t         m_graphics_queue_family{UINT32_MAX};
    VkQueue          m_graphics_queue{VK_NULL_HANDLE};
    VkQueue          m_present_queue{VK_NULL_HANDLE};

    // --- Swapchain ---
    VkSwapchainKHR           m_swapchain{VK_NULL_HANDLE};
    VkFormat                 m_swapchain_format{VK_FORMAT_UNDEFINED};
    VkExtent2D               m_swapchain_extent{0, 0};
    std::vector<VkImage>     m_swapchain_images;
    std::vector<VkImageView> m_swapchain_image_views;
    std::vector<VkFramebuffer> m_swapchain_framebuffers;

    // --- Render pass & command objects ---
    VkRenderPass     m_render_pass{VK_NULL_HANDLE};
    VkCommandPool    m_command_pool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> m_command_buffers;

    // --- Frame sync ---
    static constexpr uint32_t kMaxFramesInFlight = 2;
    std::vector<VkSemaphore> m_image_available_sems;
    std::vector<VkSemaphore> m_render_finished_sems;
    std::vector<VkFence>     m_in_flight_fences;
    uint32_t                 m_current_frame{0};
    uint32_t                 m_image_index{0};

    // --- ImGui ---
    VkDescriptorPool m_imgui_descriptor_pool{VK_NULL_HANDLE};

    // --- Textures ---
    std::unordered_map<uint64_t, TextureData> m_textures;
    uint64_t m_next_handle_id{1};
};

}  // namespace gwt::gui

#endif  // GWT_HAS_VULKAN
