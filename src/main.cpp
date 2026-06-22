/**
 * @file    main.cpp
 * @brief   Gemini Watermark Tool - Entry Point
 * @author  AllenK (Kwyshell)
 * @license MIT
 *
 * @details
 * Entry point that routes to either CLI or GUI mode based on arguments.
 *
 * Launch modes:
 *   - No arguments:           Launch GUI (if available), otherwise show help
 *   - --gui or -g:            Force GUI mode
 *   - Any other arguments:    CLI mode (original behavior)
 *   - Single file path:       CLI simple mode (in-place edit)
 */

#include "cli/cli_app.hpp"

#if defined(GWT_HAS_GUI)
#include "gui/gui_app.hpp"
#endif

#include <cstdio>
#include <cstring>
#include <string_view>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>      // _isatty
#endif

namespace {

// =============================================================================
// Platform-specific console setup
// =============================================================================

void setup_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif
}

/**
 * Check if GUI mode should be launched
 */
[[nodiscard]] bool should_launch_gui(int argc, char** argv) {
#if defined(GWT_HAS_GUI)
    // No arguments: launch GUI
    if (argc == 1) {
        return true;
    }

    // Check for explicit GUI flags. --backend=<api> selects the render
    // backend (D3D11 / Vulkan / OpenGL) and is meaningful only in the GUI,
    // so its presence implies GUI mode too.
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};
        if (arg == "--gui" || arg == "-g" || arg.starts_with("--backend")) {
            return true;
        }
    }
#else
    // Suppress unused parameter warnings
    (void)argc;
    (void)argv;
#endif

    return false;
}

#ifdef _WIN32
/**
 * Detect whether we own our console window.
 *
 * When the binary is launched via Explorer drag-drop or double-click,
 * Windows creates a transient console attached to our process; that
 * console disappears the instant we exit, taking any error message
 * printed to stdout/stderr with it. From the user's perspective this
 * looks like a silent crash even when the program reported the failure
 * cleanly.
 *
 * When launched from cmd / PowerShell / Windows Terminal / a CI shell,
 * the parent shell owns the console and it persists across our exit,
 * so the user sees the output normally.
 *
 * This helper differentiates the two by comparing the console window's
 * owning process ID to our own.
 */
bool we_own_console() {
    HWND con = GetConsoleWindow();
    if (!con) return false;
    DWORD console_pid = 0;
    GetWindowThreadProcessId(con, &console_pid);
    return console_pid == GetCurrentProcessId();
}
#endif

/**
 * Remove GUI-specific flags from argv for CLI processing
 */
void filter_gui_flags([[maybe_unused]] int& argc, [[maybe_unused]] char** argv) {
#if defined(GWT_HAS_GUI)
    int write_idx = 1;
    for (int read_idx = 1; read_idx < argc; ++read_idx) {
        std::string_view arg{argv[read_idx]};
        if (arg != "--gui" && arg != "-g") {
            argv[write_idx++] = argv[read_idx];
        }
    }
    argc = write_idx;
#endif
}

}  // anonymous namespace

int main(int argc, char** argv) {

    // Set up console for platforms
    setup_console();

    // Check if GUI mode should be launched
    if (should_launch_gui(argc, argv)) {
#if defined(GWT_HAS_GUI)
        return gwt::gui::run(argc, argv);
#else
        // GUI not compiled in, fall through to CLI
#endif
    }

    // Filter out GUI flags if any (in case --gui was combined with other args)
    filter_gui_flags(argc, argv);

    // Run CLI mode
    int exit_code = gwt::cli::run(argc, argv);

#ifdef _WIN32
    // If we were drag-dropped onto by Explorer the transient console will
    // close the moment we exit, hiding any error message. Hold the window
    // open so the user can read why we failed.
    //
    // Belt and suspenders: pause only when ALL conditions hold --
    //   - we exited with an error (no need to bother on success)
    //   - we own the console (drag-drop scenario, parent is Explorer)
    //   - stdin is an interactive TTY (a human is actually there)
    //
    // The TTY check protects MCP servers, CI runners, and any
    // programmatic invocation where stdin is piped or closed: those
    // never trigger the prompt even in the rare case where someone
    // spawns us with CREATE_NEW_CONSOLE.
    if (exit_code != 0 && we_own_console() && _isatty(_fileno(stdin))) {
        std::fputs("\nPress ENTER to close...", stderr);
        std::fflush(stderr);
        std::getchar();
    }
#endif

    return exit_code;
}
