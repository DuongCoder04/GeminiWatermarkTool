/**
 * @file    render_backend.cpp
 * @brief   Render Backend Factory Implementation
 * @author  AllenK (Kwyshell)
 * @license MIT
 */

#include "gui/backend/render_backend.hpp"
#include "gui/backend/opengl_backend.hpp"

#if defined(GWT_HAS_D3D11)
#include "gui/backend/d3d11_backend.hpp"
#endif

#if defined(GWT_HAS_VULKAN)
#include "gui/backend/vulkan_backend.hpp"
#endif

#include <spdlog/spdlog.h>

namespace gwt::gui {

std::unique_ptr<IRenderBackend> create_backend(BackendType type) {
    // Auto mode: platform-specific preference order.
    //   Windows: D3D11 > Vulkan > OpenGL
    //   Linux:   Vulkan > OpenGL
    //   macOS:   OpenGL > Vulkan   (Vulkan needs a user-installed MoltenVK)
    if (type == BackendType::Auto) {
#if defined(_WIN32)
        // D3D11 first: besides good VM/RDP behaviour, it falls back to the
        // WARP software rasterizer, so the UI still renders with NO GPU --
        // Vulkan and OpenGL have no such guaranteed software path on Windows.
#  if defined(GWT_HAS_D3D11)
        if (is_backend_available(BackendType::D3D11)) {
            spdlog::info("Auto-selecting D3D11 backend");
            return std::make_unique<D3D11Backend>();
        }
        spdlog::debug("D3D11 unavailable, trying Vulkan/OpenGL");
#  endif
#  if defined(GWT_HAS_VULKAN)
        if (is_backend_available(BackendType::Vulkan)) {
            spdlog::info("Auto-selecting Vulkan backend");
            return std::make_unique<VulkanBackend>();
        }
        spdlog::debug("Vulkan unavailable, falling back to OpenGL");
#  endif
        type = BackendType::OpenGL;

#elif defined(__APPLE__)
        // macOS default is OpenGL; Vulkan (via MoltenVK) isn't present unless
        // the user installs it, so it stays an explicit opt-in (--backend=vulkan).
        spdlog::info("Auto-selecting OpenGL backend (macOS default)");
        type = BackendType::OpenGL;

#else
        // Linux / other Unix: Vulkan > OpenGL.
#  if defined(GWT_HAS_VULKAN)
        if (is_backend_available(BackendType::Vulkan)) {
            spdlog::info("Auto-selecting Vulkan backend");
            return std::make_unique<VulkanBackend>();
        }
        spdlog::debug("Vulkan unavailable, falling back to OpenGL");
#  endif
        type = BackendType::OpenGL;
#endif
    }

    // Create specific backend
    switch (type) {
        case BackendType::OpenGL:
            spdlog::info("Creating OpenGL backend");
            return std::make_unique<OpenGLBackend>();

#if defined(GWT_HAS_D3D11)
        case BackendType::D3D11:
            spdlog::info("Creating D3D11 backend");
            return std::make_unique<D3D11Backend>();
#endif

#if defined(GWT_HAS_VULKAN)
        case BackendType::Vulkan:
            spdlog::info("Creating Vulkan backend");
            return std::make_unique<VulkanBackend>();
#endif

        default:
            spdlog::error("Unknown backend type requested");
            return nullptr;
    }
}

bool is_backend_available(BackendType type) noexcept {
    switch (type) {
        case BackendType::OpenGL:
            // OpenGL is always available (compiled in)
            return true;

#if defined(GWT_HAS_D3D11)
        case BackendType::D3D11:
            // Check if D3D11 runtime is available
            return D3D11Backend::is_available();
#endif

#if defined(GWT_HAS_VULKAN)
        case BackendType::Vulkan:
            // Check if Vulkan runtime is available
            return VulkanBackend::is_available();
#endif

        case BackendType::Auto:
            // Auto is always "available" (will fall back)
            return true;

        default:
            return false;
    }
}

}  // namespace gwt::gui
