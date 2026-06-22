/**
 * @file    vulkan_backend.cpp
 * @brief   Vulkan 1.2 Render Backend Implementation
 * @license MIT
 *
 * @details
 * Minimal but working Vulkan backend that mirrors the shape of the
 * OpenGL and D3D11 backends. Only clears the screen and renders ImGui -
 * intentionally keeps features small so the template stays readable.
 *
 * Not implemented (left as template extension points):
 *   - Mipmapped / compressed texture upload
 *   - MSAA resolve
 *   - Compute queue
 *   - Validation layer in Release builds
 */

#if defined(GWT_HAS_VULKAN)

#include "gui/backend/vulkan_backend.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>

namespace gwt::gui {

namespace {

#if defined(DEBUG) || defined(_DEBUG)
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

constexpr std::array<const char*, 1> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

constexpr std::array<const char*, 1> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        spdlog::error("Vulkan: {} failed with VkResult={}", what, static_cast<int>(result));
    }
}

bool layer_available(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) return true;
    }
    return false;
}

}  // anonymous namespace

// =============================================================================
// Static Availability Check
// =============================================================================

bool VulkanBackend::is_available() noexcept {
    if (!SDL_Vulkan_LoadLibrary(nullptr)) {
        spdlog::debug("Vulkan: SDL_Vulkan_LoadLibrary failed: {}", SDL_GetError());
        return false;
    }

    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = APP_NAME;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName        = "cpp-modern-template";
    app_info.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance test_instance{VK_NULL_HANDLE};
    const VkResult result = vkCreateInstance(&create_info, nullptr, &test_instance);

    bool ok = false;
    if (result == VK_SUCCESS && test_instance) {
        vkDestroyInstance(test_instance, nullptr);
        ok = true;
    }

    SDL_Vulkan_UnloadLibrary();
    return ok;
}

// =============================================================================
// Lifecycle
// =============================================================================

VulkanBackend::~VulkanBackend() {
    if (m_initialized) {
        shutdown();
    }
}

bool VulkanBackend::init(SDL_Window* window) {
    if (m_initialized) return true;
    if (!window) {
        set_error(BackendError::InitFailed);
        return false;
    }
    m_window = window;

    SDL_GetWindowSizeInPixels(window, &m_window_width, &m_window_height);

    if (!create_instance())          { set_error(BackendError::InitFailed);          return false; }
    if (!create_surface())           { set_error(BackendError::WindowCreationFailed);return false; }
    if (!pick_physical_device())     { set_error(BackendError::InitFailed);          return false; }
    if (!create_logical_device())    { set_error(BackendError::InitFailed);          return false; }
    if (!create_swapchain())         { set_error(BackendError::InitFailed);          return false; }
    if (!create_image_views())       { set_error(BackendError::InitFailed);          return false; }
    if (!create_render_pass())       { set_error(BackendError::InitFailed);          return false; }
    if (!create_framebuffers())      { set_error(BackendError::InitFailed);          return false; }
    if (!create_command_pool())      { set_error(BackendError::InitFailed);          return false; }
    if (!create_command_buffers())   { set_error(BackendError::InitFailed);          return false; }
    if (!create_sync_objects())      { set_error(BackendError::InitFailed);          return false; }
    if (!create_descriptor_pool())   { set_error(BackendError::InitFailed);          return false; }

    m_initialized = true;
    clear_error();
    spdlog::info("Vulkan backend initialized ({}x{}, format={})",
                 m_swapchain_extent.width, m_swapchain_extent.height,
                 static_cast<int>(m_swapchain_format));
    return true;
}

void VulkanBackend::shutdown() {
    if (!m_initialized) return;

    if (m_device) {
        vkDeviceWaitIdle(m_device);
    }

    if (m_imgui_descriptor_pool) {
        vkDestroyDescriptorPool(m_device, m_imgui_descriptor_pool, nullptr);
        m_imgui_descriptor_pool = VK_NULL_HANDLE;
    }

    for (size_t i = 0; i < m_in_flight_fences.size(); ++i) {
        if (m_in_flight_fences[i])      vkDestroyFence(m_device, m_in_flight_fences[i], nullptr);
        if (m_image_available_sems[i])  vkDestroySemaphore(m_device, m_image_available_sems[i], nullptr);
        if (m_render_finished_sems[i])  vkDestroySemaphore(m_device, m_render_finished_sems[i], nullptr);
    }
    m_in_flight_fences.clear();
    m_image_available_sems.clear();
    m_render_finished_sems.clear();

    cleanup_swapchain();

    if (m_command_pool) {
        vkDestroyCommandPool(m_device, m_command_pool, nullptr);
        m_command_pool = VK_NULL_HANDLE;
    }
    if (m_render_pass) {
        vkDestroyRenderPass(m_device, m_render_pass, nullptr);
        m_render_pass = VK_NULL_HANDLE;
    }
    if (m_device) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
    if (m_surface) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
    if (m_instance) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_textures.clear();
    m_window      = nullptr;
    m_initialized = false;
}

// =============================================================================
// Instance / Surface / Device
// =============================================================================

bool VulkanBackend::create_instance() {
    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = APP_NAME;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName        = "cpp-modern-template";
    app_info.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_2;

    // Collect extensions required by SDL
    Uint32 sdl_ext_count = 0;
    const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);
    std::vector<const char*> extensions(sdl_exts, sdl_exts + sdl_ext_count);

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app_info;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    if constexpr (kEnableValidation) {
        if (layer_available(kValidationLayers[0])) {
            ci.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
            ci.ppEnabledLayerNames = kValidationLayers.data();
            spdlog::debug("Vulkan: validation layer enabled");
        }
    }  // if constexpr (kEnableValidation)

    const VkResult r = vkCreateInstance(&ci, nullptr, &m_instance);
    if (r != VK_SUCCESS) {
        spdlog::error("vkCreateInstance failed: {}", static_cast<int>(r));
        return false;
    }
    return true;
}

bool VulkanBackend::create_surface() {
    if (!SDL_Vulkan_CreateSurface(m_window, m_instance, nullptr, &m_surface)) {
        spdlog::error("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        return false;
    }
    return true;
}

bool VulkanBackend::pick_physical_device() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) {
        spdlog::error("Vulkan: no physical devices found");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    // Simple heuristic: pick the first device that supports graphics +
    // presentation + swapchain extension. Prefer discrete GPUs.
    auto score_device = [this](VkPhysicalDevice dev) -> int {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, queues.data());

        bool has_graphics_present = false;
        for (uint32_t i = 0; i < qcount; ++i) {
            if (!(queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 present_ok = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &present_ok);
            if (present_ok) { has_graphics_present = true; break; }
        }
        if (!has_graphics_present) return -1;

        return props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 100;
    };

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int best_score = -1;
    for (auto dev : devices) {
        const int s = score_device(dev);
        if (s > best_score) { best = dev; best_score = s; }
    }

    if (best == VK_NULL_HANDLE) {
        spdlog::error("Vulkan: no suitable physical device");
        return false;
    }
    m_physical_device = best;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physical_device, &props);
    spdlog::info("Vulkan: selected GPU: {}", props.deviceName);

    // Pick graphics queue family
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> queues(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &qcount, queues.data());
    for (uint32_t i = 0; i < qcount; ++i) {
        if (!(queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        VkBool32 present_ok = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(m_physical_device, i, m_surface, &present_ok);
        if (present_ok) {
            m_graphics_queue_family = i;
            break;
        }
    }
    return m_graphics_queue_family != UINT32_MAX;
}

bool VulkanBackend::create_logical_device() {
    const float priority = 1.0f;

    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_graphics_queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.pEnabledFeatures        = &features;
    dci.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
    dci.ppEnabledExtensionNames = kDeviceExtensions.data();

    const VkResult r = vkCreateDevice(m_physical_device, &dci, nullptr, &m_device);
    if (r != VK_SUCCESS) {
        spdlog::error("vkCreateDevice failed: {}", static_cast<int>(r));
        return false;
    }
    vkGetDeviceQueue(m_device, m_graphics_queue_family, 0, &m_graphics_queue);
    m_present_queue = m_graphics_queue;  // same family
    return true;
}

// =============================================================================
// Swapchain
// =============================================================================

bool VulkanBackend::create_swapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device, m_surface, &caps);

    // Format selection
    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &fmt_count, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    m_swapchain_format = chosen.format;

    // Extent
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        m_swapchain_extent = caps.currentExtent;
    } else {
        int w = m_window_width, h = m_window_height;
        SDL_GetWindowSizeInPixels(m_window, &w, &h);
        m_swapchain_extent.width  = std::clamp(static_cast<uint32_t>(w),
                                               caps.minImageExtent.width,
                                               caps.maxImageExtent.width);
        m_swapchain_extent.height = std::clamp(static_cast<uint32_t>(h),
                                               caps.minImageExtent.height,
                                               caps.maxImageExtent.height);
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = m_surface;
    sci.minImageCount    = image_count;
    sci.imageFormat      = chosen.format;
    sci.imageColorSpace  = chosen.colorSpace;
    sci.imageExtent      = m_swapchain_extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped          = VK_TRUE;
    sci.oldSwapchain     = VK_NULL_HANDLE;

    const VkResult r = vkCreateSwapchainKHR(m_device, &sci, nullptr, &m_swapchain);
    if (r != VK_SUCCESS) {
        spdlog::error("vkCreateSwapchainKHR failed: {}", static_cast<int>(r));
        return false;
    }

    uint32_t actual = 0;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual, nullptr);
    m_swapchain_images.resize(actual);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual, m_swapchain_images.data());
    return true;
}

bool VulkanBackend::create_image_views() {
    m_swapchain_image_views.resize(m_swapchain_images.size());
    for (size_t i = 0; i < m_swapchain_images.size(); ++i) {
        VkImageViewCreateInfo ivi{};
        ivi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivi.image                           = m_swapchain_images[i];
        ivi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format                          = m_swapchain_format;
        ivi.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ivi.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ivi.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ivi.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        ivi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ivi.subresourceRange.baseMipLevel   = 0;
        ivi.subresourceRange.levelCount     = 1;
        ivi.subresourceRange.baseArrayLayer = 0;
        ivi.subresourceRange.layerCount     = 1;
        const VkResult r = vkCreateImageView(m_device, &ivi, nullptr, &m_swapchain_image_views[i]);
        if (r != VK_SUCCESS) {
            spdlog::error("vkCreateImageView failed: {}", static_cast<int>(r));
            return false;
        }
    }
    return true;
}

bool VulkanBackend::create_render_pass() {
    VkAttachmentDescription color{};
    color.format         = m_swapchain_format;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &color;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;

    const VkResult r = vkCreateRenderPass(m_device, &rpci, nullptr, &m_render_pass);
    if (r != VK_SUCCESS) {
        spdlog::error("vkCreateRenderPass failed: {}", static_cast<int>(r));
        return false;
    }
    return true;
}

bool VulkanBackend::create_framebuffers() {
    m_swapchain_framebuffers.resize(m_swapchain_image_views.size());
    for (size_t i = 0; i < m_swapchain_image_views.size(); ++i) {
        VkFramebufferCreateInfo fbi{};
        fbi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbi.renderPass      = m_render_pass;
        fbi.attachmentCount = 1;
        fbi.pAttachments    = &m_swapchain_image_views[i];
        fbi.width           = m_swapchain_extent.width;
        fbi.height          = m_swapchain_extent.height;
        fbi.layers          = 1;
        const VkResult r = vkCreateFramebuffer(m_device, &fbi, nullptr,
                                               &m_swapchain_framebuffers[i]);
        if (r != VK_SUCCESS) {
            spdlog::error("vkCreateFramebuffer failed: {}", static_cast<int>(r));
            return false;
        }
    }
    return true;
}

bool VulkanBackend::create_command_pool() {
    VkCommandPoolCreateInfo cpi{};
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = m_graphics_queue_family;
    const VkResult r = vkCreateCommandPool(m_device, &cpi, nullptr, &m_command_pool);
    if (r != VK_SUCCESS) {
        spdlog::error("vkCreateCommandPool failed: {}", static_cast<int>(r));
        return false;
    }
    return true;
}

bool VulkanBackend::create_command_buffers() {
    m_command_buffers.resize(kMaxFramesInFlight);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_command_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kMaxFramesInFlight;
    const VkResult r = vkAllocateCommandBuffers(m_device, &ai, m_command_buffers.data());
    if (r != VK_SUCCESS) {
        spdlog::error("vkAllocateCommandBuffers failed: {}", static_cast<int>(r));
        return false;
    }
    return true;
}

bool VulkanBackend::create_sync_objects() {
    m_image_available_sems.resize(kMaxFramesInFlight);
    m_render_finished_sems.resize(kMaxFramesInFlight);
    m_in_flight_fences.resize(kMaxFramesInFlight);

    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (vkCreateSemaphore(m_device, &si, nullptr, &m_image_available_sems[i]) != VK_SUCCESS ||
            vkCreateSemaphore(m_device, &si, nullptr, &m_render_finished_sems[i]) != VK_SUCCESS ||
            vkCreateFence    (m_device, &fi, nullptr, &m_in_flight_fences[i])     != VK_SUCCESS) {
            spdlog::error("Failed to create per-frame sync objects");
            return false;
        }
    }
    return true;
}

bool VulkanBackend::create_descriptor_pool() {
    // Large enough for ImGui font atlas + a few extra textures.
    std::array<VkDescriptorPoolSize, 1> pool_sizes = {{
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 }
    }};

    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets       = 1024;
    pci.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pci.pPoolSizes    = pool_sizes.data();

    const VkResult r = vkCreateDescriptorPool(m_device, &pci, nullptr, &m_imgui_descriptor_pool);
    if (r != VK_SUCCESS) {
        spdlog::error("vkCreateDescriptorPool failed: {}", static_cast<int>(r));
        return false;
    }
    return true;
}

void VulkanBackend::cleanup_swapchain() {
    if (!m_device) return;

    for (auto fb : m_swapchain_framebuffers) {
        if (fb) vkDestroyFramebuffer(m_device, fb, nullptr);
    }
    m_swapchain_framebuffers.clear();

    for (auto view : m_swapchain_image_views) {
        if (view) vkDestroyImageView(m_device, view, nullptr);
    }
    m_swapchain_image_views.clear();

    if (m_swapchain) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
    m_swapchain_images.clear();
}

bool VulkanBackend::recreate_swapchain() {
    if (!m_device) return false;
    vkDeviceWaitIdle(m_device);

    cleanup_swapchain();

    if (!create_swapchain())    return false;
    if (!create_image_views())  return false;
    if (!create_framebuffers()) return false;
    return true;
}

// =============================================================================
// ImGui Integration
// =============================================================================

void VulkanBackend::imgui_init() {
    ImGui_ImplSDL3_InitForVulkan(m_window);

    ImGui_ImplVulkan_InitInfo info{};
    info.Instance        = m_instance;
    info.PhysicalDevice  = m_physical_device;
    info.Device          = m_device;
    info.QueueFamily     = m_graphics_queue_family;
    info.Queue           = m_graphics_queue;
    info.PipelineCache   = VK_NULL_HANDLE;
    info.DescriptorPool  = m_imgui_descriptor_pool;
    info.Subpass         = 0;
    info.MinImageCount   = 2;
    info.ImageCount      = static_cast<uint32_t>(m_swapchain_images.size());
    info.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    info.Allocator       = nullptr;
    info.CheckVkResultFn = nullptr;
    info.RenderPass      = m_render_pass;

    ImGui_ImplVulkan_Init(&info);
    ImGui_ImplVulkan_CreateFontsTexture();
}

void VulkanBackend::imgui_shutdown() {
    if (m_device) vkDeviceWaitIdle(m_device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
}

void VulkanBackend::imgui_new_frame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
}

void VulkanBackend::imgui_render() {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                    m_command_buffers[m_current_frame]);
}

// =============================================================================
// Frame Management
// =============================================================================

void VulkanBackend::begin_frame() {
    if (m_swapchain_dirty) {
        recreate_swapchain();
        m_swapchain_dirty = false;
    }

    vkWaitForFences(m_device, 1, &m_in_flight_fences[m_current_frame], VK_TRUE, UINT64_MAX);

    VkResult acq = vkAcquireNextImageKHR(m_device,
                                         m_swapchain,
                                         UINT64_MAX,
                                         m_image_available_sems[m_current_frame],
                                         VK_NULL_HANDLE,
                                         &m_image_index);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        m_swapchain_dirty = true;
        return;
    }
    check(acq == VK_SUBOPTIMAL_KHR ? VK_SUCCESS : acq, "vkAcquireNextImageKHR");

    vkResetFences(m_device, 1, &m_in_flight_fences[m_current_frame]);

    VkCommandBuffer cb = m_command_buffers[m_current_frame];
    vkResetCommandBuffer(cb, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);

    VkClearValue clear{};
    clear.color = { { 0.1f, 0.1f, 0.1f, 1.0f } };

    VkRenderPassBeginInfo rbi{};
    rbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass        = m_render_pass;
    rbi.framebuffer       = m_swapchain_framebuffers[m_image_index];
    rbi.renderArea.offset = {0, 0};
    rbi.renderArea.extent = m_swapchain_extent;
    rbi.clearValueCount   = 1;
    rbi.pClearValues      = &clear;
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
}

void VulkanBackend::end_frame() {
    VkCommandBuffer cb = m_command_buffers[m_current_frame];
    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
}

void VulkanBackend::present() {
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_sems[]   = { m_image_available_sems[m_current_frame] };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore signal_sems[] = { m_render_finished_sems[m_current_frame] };

    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = wait_sems;
    si.pWaitDstStageMask    = wait_stages;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &m_command_buffers[m_current_frame];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = signal_sems;

    check(vkQueueSubmit(m_graphics_queue, 1, &si, m_in_flight_fences[m_current_frame]),
          "vkQueueSubmit");

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = signal_sems;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &m_image_index;

    const VkResult pr = vkQueuePresentKHR(m_present_queue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        m_swapchain_dirty = true;
    } else {
        check(pr, "vkQueuePresentKHR");
    }

    m_current_frame = (m_current_frame + 1) % kMaxFramesInFlight;
}

void VulkanBackend::on_resize(int width, int height) {
    m_window_width  = width;
    m_window_height = height;
    m_swapchain_dirty = true;
}

// =============================================================================
// Texture upload helpers
// =============================================================================

namespace {

// Normalize any supported source layout into tightly-packed RGBA8.
// Returns a vector of size w*h*4. The GUI only ever sends RGBA8, but the
// D3D11 backend handles all four formats so we mirror that for parity.
std::vector<uint8_t> to_rgba8(const TextureDesc& desc, std::span<const uint8_t> data) {
    const size_t pixel_count = static_cast<size_t>(desc.width) * desc.height;
    std::vector<uint8_t> out(pixel_count * 4, 0xFF);  // default opaque
    if (data.empty()) return out;

    switch (desc.format) {
        case TextureFormat::RGBA8: {
            const size_t n = std::min(data.size(), out.size());
            std::memcpy(out.data(), data.data(), n);
            break;
        }
        case TextureFormat::BGRA8: {
            const size_t n = std::min<size_t>(pixel_count, data.size() / 4);
            for (size_t i = 0; i < n; ++i) {
                out[i * 4 + 0] = data[i * 4 + 2];  // R <- B
                out[i * 4 + 1] = data[i * 4 + 1];  // G
                out[i * 4 + 2] = data[i * 4 + 0];  // B <- R
                out[i * 4 + 3] = data[i * 4 + 3];  // A
            }
            break;
        }
        case TextureFormat::RGB8: {
            const size_t n = std::min<size_t>(pixel_count, data.size() / 3);
            for (size_t i = 0; i < n; ++i) {
                out[i * 4 + 0] = data[i * 3 + 0];
                out[i * 4 + 1] = data[i * 3 + 1];
                out[i * 4 + 2] = data[i * 3 + 2];
                out[i * 4 + 3] = 0xFF;
            }
            break;
        }
        case TextureFormat::BGR8: {
            const size_t n = std::min<size_t>(pixel_count, data.size() / 3);
            for (size_t i = 0; i < n; ++i) {
                out[i * 4 + 0] = data[i * 3 + 2];  // R <- B
                out[i * 4 + 1] = data[i * 3 + 1];  // G
                out[i * 4 + 2] = data[i * 3 + 0];  // B <- R
                out[i * 4 + 3] = 0xFF;
            }
            break;
        }
    }
    return out;
}

}  // anonymous namespace

uint32_t VulkanBackend::find_memory_type(uint32_t type_filter,
                                         VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(m_physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        const bool type_ok = (type_filter & (1u << i)) != 0;
        const bool props_ok =
            (mem_props.memoryTypes[i].propertyFlags & props) == props;
        if (type_ok && props_ok) return i;
    }
    spdlog::error("Vulkan: no suitable memory type (filter=0x{:x})", type_filter);
    return 0;
}

bool VulkanBackend::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags props,
                                  VkBuffer& buffer, VkDeviceMemory& memory) const {
    VkBufferCreateInfo info{};
    info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size        = size;
    info.usage       = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &info, nullptr, &buffer) != VK_SUCCESS) {
        spdlog::error("Vulkan: vkCreateBuffer failed");
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_device, buffer, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
    if (vkAllocateMemory(m_device, &alloc, nullptr, &memory) != VK_SUCCESS) {
        spdlog::error("Vulkan: vkAllocateMemory (buffer) failed");
        vkDestroyBuffer(m_device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(m_device, buffer, memory, 0);
    return true;
}

VkCommandBuffer VulkanBackend::begin_single_time() const {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool        = m_command_pool;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void VulkanBackend::end_single_time(VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;

    // Synchronous: a transient upload at load time, not on the render hot path.
    vkQueueSubmit(m_graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphics_queue);
    vkFreeCommandBuffers(m_device, m_command_pool, 1, &cmd);
}

void VulkanBackend::upload_rgba_to_image(VkImage image, const void* pixels,
                                         uint32_t w, uint32_t h) const {
    const VkDeviceSize image_size = static_cast<VkDeviceSize>(w) * h * 4;

    VkBuffer       staging        = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    if (!create_buffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       staging, staging_memory)) {
        return;
    }

    void* mapped = nullptr;
    vkMapMemory(m_device, staging_memory, 0, image_size, 0, &mapped);
    std::memcpy(mapped, pixels, static_cast<size_t>(image_size));
    vkUnmapMemory(m_device, staging_memory);

    VkCommandBuffer cmd = begin_single_time();

    // UNDEFINED -> TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask       = 0;
    barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset       = {0, 0, 0};
    region.imageExtent       = {w, h, 1};
    vkCmdCopyBufferToImage(cmd, staging, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    end_single_time(cmd);

    vkDestroyBuffer(m_device, staging, nullptr);
    vkFreeMemory(m_device, staging_memory, nullptr);
}

// =============================================================================
// Texture Operations
// =============================================================================

TextureHandle
VulkanBackend::create_texture(const TextureDesc& desc, std::span<const uint8_t> data) {
    if (!m_initialized || m_device == VK_NULL_HANDLE) {
        spdlog::error("VulkanBackend::create_texture before initialization");
        set_error(BackendError::TextureCreationFailed);
        return TextureHandle{};
    }
    if (desc.width == 0 || desc.height == 0) {
        spdlog::error("VulkanBackend::create_texture: zero-sized texture");
        set_error(BackendError::TextureCreationFailed);
        return TextureHandle{};
    }

    TextureData tex{};
    tex.desc = desc;
    constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

    // 1. Device-local image (sampled + transfer destination).
    VkImageCreateInfo image_info{};
    image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType     = VK_IMAGE_TYPE_2D;
    image_info.extent        = {desc.width, desc.height, 1};
    image_info.mipLevels     = 1;
    image_info.arrayLayers   = 1;
    image_info.format        = kFormat;
    image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(m_device, &image_info, nullptr, &tex.image) != VK_SUCCESS) {
        spdlog::error("Vulkan: vkCreateImage failed");
        set_error(BackendError::TextureCreationFailed);
        return TextureHandle{};
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_device, tex.image, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex =
        find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(m_device, &alloc, nullptr, &tex.memory) != VK_SUCCESS) {
        spdlog::error("Vulkan: vkAllocateMemory (image) failed");
        vkDestroyImage(m_device, tex.image, nullptr);
        set_error(BackendError::TextureCreationFailed);
        return TextureHandle{};
    }
    vkBindImageMemory(m_device, tex.image, tex.memory, 0);

    // 2. Upload pixels (normalized to RGBA8) via staging buffer.
    const std::vector<uint8_t> rgba = to_rgba8(desc, data);
    upload_rgba_to_image(tex.image, rgba.data(), desc.width, desc.height);

    // 3. Image view.
    VkImageViewCreateInfo view_info{};
    view_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image    = tex.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format   = kFormat;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(m_device, &view_info, nullptr, &tex.view) != VK_SUCCESS) {
        spdlog::error("Vulkan: vkCreateImageView failed");
        vkDestroyImage(m_device, tex.image, nullptr);
        vkFreeMemory(m_device, tex.memory, nullptr);
        set_error(BackendError::TextureCreationFailed);
        return TextureHandle{};
    }

    // 4. Sampler (linear filtering, clamp to edge — matches the other backends).
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter    = VK_FILTER_LINEAR;
    sampler_info.minFilter    = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sampler_info.maxLod       = 1.0f;
    if (vkCreateSampler(m_device, &sampler_info, nullptr, &tex.sampler) != VK_SUCCESS) {
        spdlog::error("Vulkan: vkCreateSampler failed");
        vkDestroyImageView(m_device, tex.view, nullptr);
        vkDestroyImage(m_device, tex.image, nullptr);
        vkFreeMemory(m_device, tex.memory, nullptr);
        set_error(BackendError::TextureCreationFailed);
        return TextureHandle{};
    }

    // 5. Register an ImGui-drawable descriptor set; this IS the ImTextureID.
    tex.descriptor_set = ImGui_ImplVulkan_AddTexture(
        tex.sampler, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    const TextureHandle handle{m_next_handle_id++};
    m_textures[handle.id] = tex;
    return handle;
}

void VulkanBackend::update_texture(TextureHandle handle, std::span<const uint8_t> data) {
    auto it = m_textures.find(handle.id);
    if (it == m_textures.end()) {
        spdlog::warn("VulkanBackend::update_texture: unknown handle {}", handle.id);
        return;
    }
    const TextureData& tex = it->second;
    if (tex.image == VK_NULL_HANDLE) return;

    // Full re-upload. The image/view/sampler/descriptor stay valid, so the
    // ImTextureID is unchanged; only the device-local pixels are replaced.
    const std::vector<uint8_t> rgba = to_rgba8(tex.desc, data);
    upload_rgba_to_image(tex.image, rgba.data(), tex.desc.width, tex.desc.height);
}

void VulkanBackend::destroy_texture(TextureHandle handle) {
    auto it = m_textures.find(handle.id);
    if (it == m_textures.end()) return;
    TextureData& tex = it->second;

    // The device may still be referencing these from an in-flight frame.
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
        if (tex.descriptor_set != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(tex.descriptor_set);
        }
        if (tex.sampler != VK_NULL_HANDLE) vkDestroySampler(m_device, tex.sampler, nullptr);
        if (tex.view != VK_NULL_HANDLE)    vkDestroyImageView(m_device, tex.view, nullptr);
        if (tex.image != VK_NULL_HANDLE)   vkDestroyImage(m_device, tex.image, nullptr);
        if (tex.memory != VK_NULL_HANDLE)  vkFreeMemory(m_device, tex.memory, nullptr);
    }
    m_textures.erase(it);
}

void* VulkanBackend::get_imgui_texture_id(TextureHandle handle) const {
    auto it = m_textures.find(handle.id);
    if (it == m_textures.end()) return nullptr;
    return reinterpret_cast<void*>(it->second.descriptor_set);
}

// =============================================================================
// Backend Info
// =============================================================================

std::string_view VulkanBackend::name() const noexcept {
    return "Vulkan 1.2";
}

}  // namespace gwt::gui

#endif  // GWT_HAS_VULKAN
