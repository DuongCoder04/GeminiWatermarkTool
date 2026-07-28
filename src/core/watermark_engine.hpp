#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <optional>
#include <filesystem>
#include <atomic>

namespace gwt {

/**
 * Watermark size mode based on image dimensions
 */
enum class WatermarkSize {
    Small,   // small canonical size
    Large,   // large canonical size
};

/**
 * Watermark variant.
 *
 * Two co-existing watermark profiles are recognised. V1 covers outputs
 * from Gemini versions before 3.5; V2 covers outputs from 3.5 onward.
 * Detection runs against the active variant only; the variant is chosen
 * explicitly via the CLI --legacy flag or the GUI Legacy checkbox.
 */
enum class WatermarkVariant {
    V1,   // legacy profile
    V2,   // current profile
};

/**
 * Watermark detection result
 */
struct DetectionResult {
    bool detected;           // Whether watermark was detected
    float confidence;        // Detection confidence (0.0 - 1.0)
    cv::Rect region;         // Detected watermark region
    WatermarkSize size;      // Detected watermark size
    WatermarkVariant variant{WatermarkVariant::V2};  // Which profile matched

    // Debug info
    float spatial_score;     // Stage 1: Spatial NCC score
    float gradient_score;    // Stage 2: Gradient NCC score
    float variance_score;    // Stage 3: Variance analysis score
};

/**
 * Guided detection result (multi-scale search within user-specified region)
 */
struct GuidedDetectionResult {
    bool found{false};           // Whether a match was found
    float confidence{0.0f};      // Size-adjusted score (for decision making)
    float raw_ncc{0.0f};         // Raw NCC score (before size adjustment)
    cv::Rect match_rect;         // Best match position and size (image coords)
    int detected_size{0};        // Detected watermark size in pixels
    bool was_cancelled{false};   // Search was cancelled by user

    // Search stats
    int scales_searched{0};      // Number of scales tested
    int total_scales{0};         // Total scales planned
};

/**
 * Watermark position configuration
 */
struct WatermarkPosition {
    int margin_right;   // Distance from right edge
    int margin_bottom;  // Distance from bottom edge
    int logo_size;      // 48 or 96

    // Get top-left position for a given image size
    cv::Point get_position(int image_width, int image_height) const {
        return cv::Point(
            image_width - margin_right - logo_size,
            image_height - margin_bottom - logo_size
        );
    }
};

/**
 * Get the watermark configuration for a given image size and variant.
 *
 * Each variant has its own logo size and margin from the bottom-right
 * corner. The two-argument overload defaults to V2 (current profile);
 * callers that need the legacy profile must request it explicitly.
 */
WatermarkPosition get_watermark_config(int image_width, int image_height,
                                       WatermarkVariant variant);

/// Convenience overload: defaults to V2 (current profile).
inline WatermarkPosition get_watermark_config(int image_width, int image_height) {
    return get_watermark_config(image_width, image_height, WatermarkVariant::V2);
}

/**
 * Determine watermark size mode from image dimensions
 */
WatermarkSize get_watermark_size(int image_width, int image_height);

/**
 * Inpaint method for residual cleanup after reverse alpha blend
 */
enum class InpaintMethod {
    GAUSSIAN,   // Soft-blend: continuous gradient mask + Gaussian blur (recommended)
    TELEA,      // OpenCV: Fast Marching Method (Telea, 2004)
    NS,         // OpenCV: Navier-Stokes (Bertalmio et al., 2001)
#ifdef GWT_HAS_AI_DENOISE
    AI_DENOISE  // FDnCNN: Neural network residual denoising (NCNN + Vulkan GPU)
#endif
};

/**
 * Main watermark engine class
 *
 * Uses background captures to dynamically calculate alpha maps.
 * No pre-processed masks needed - just the original captures.
 *
 * Math:
 *   Gemini adds watermark: result = alpha * logo + (1 - alpha) * original
 *   To remove: original = (result - alpha * 255) / (1 - alpha)
 */
class WatermarkEngine {
public:
    /**
     * Initialize the engine with background captures from files
     *
     * These are the raw captures from Gemini on pure background.
     * The engine will dynamically calculate alpha maps from them.
     *
     * @param bg_small  Path to 48x48 background capture
     * @param bg_large  Path to 96x96 background capture
     * @param logo_value      The logo brightness (default: 255.0 = white)
     */
    WatermarkEngine(
        const std::filesystem::path& bg_small,
        const std::filesystem::path& bg_large,
        float logo_value = 255.0f
    );

    /**
     * Initialize the engine with embedded PNG data (standalone mode)
     *
     * V1 (legacy) and V2 (current) profile assets are both loaded; the
     * detector picks the best-matching one per image at runtime.
     *
     * @param v1_small, v1_small_size  V1 small BG capture PNG bytes
     * @param v1_large, v1_large_size  V1 large BG capture PNG bytes
     * @param v2_small, v2_small_size  V2 small BG capture PNG bytes
     * @param v2_large, v2_large_size  V2 large BG capture PNG bytes
     * @param logo_value  The logo brightness (default: 255.0 = white)
     */
    WatermarkEngine(
        const unsigned char* v1_small, size_t v1_small_size,
        const unsigned char* v1_large, size_t v1_large_size,
        const unsigned char* v2_small, size_t v2_small_size,
        const unsigned char* v2_large, size_t v2_large_size,
        float logo_value = 255.0f
    );

    /**
     * Two-pair convenience constructor (V1-only). Retained for callers
     * that have not been updated to supply V2 assets; V2 detection will
     * be a no-op for engines built this way.
     */
    WatermarkEngine(
        const unsigned char* png_data_small, size_t png_size_small,
        const unsigned char* png_data_large, size_t png_size_large,
        float logo_value = 255.0f
    );

    /**
     * Detect watermark in an image using alpha map correlation
     *
     * Three-stage detection algorithm:
     *   Stage 1: Spatial NCC - Normalized cross-correlation with alpha map
     *   Stage 2: Gradient NCC - Edge signature correlation
     *   Stage 3: Variance Analysis - Texture dampening detection
     *
     * @param image          The image to analyze
     * @param force_size     Force a specific watermark size (auto-detect if nullopt)
     * @param force_variant  Force a specific variant (try both if nullopt).
     *                       When both variants are tried, the higher-confidence
     *                       match wins and is reflected in the returned variant.
     * @return               Detection result with confidence, region, and variant
     */
    DetectionResult detect_watermark(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt,
        std::optional<WatermarkVariant> force_variant = std::nullopt
    ) const;

    /**
     * Guided multi-scale watermark detection within a user-specified region
     *
     * Uses coarse-to-fine NCC template matching:
     *   Phase 1 (coarse): Large scale/position steps, find top candidates
     *   Phase 2 (fine):   Refine around top candidates with pixel precision
     *
     * The search resizes the alpha map template to multiple scales and
     * performs NCC matching at each scale within the search region.
     *
     * @param image         The image to search
     * @param search_rect   User-drawn search region (image coordinates)
     * @param cancel_flag   Atomic flag to request cancellation (optional)
     * @param min_size      Minimum watermark size to search (default: 32)
     * @param max_size      Maximum watermark size to search (default: 160)
     * @return              Guided detection result with best match
     */
    GuidedDetectionResult guided_detect(
        const cv::Mat& image,
        const cv::Rect& search_rect,
        std::atomic<bool>* cancel_flag = nullptr,
        int min_size = 32,
        int max_size = 160
    ) const;

    /**
     * Remove watermark from an image
     *
     * @param image          The image to process (will be modified in-place)
     * @param force_size     Force a specific watermark size (auto-detect if nullopt)
     * @param force_variant  Force a specific profile (auto-detect if nullopt)
     */
    void remove_watermark(
        cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt,
        std::optional<WatermarkVariant> force_variant = std::nullopt
    );

    /**
     * Remove watermark from a custom region with interpolated alpha map
     *
     * @param image          The image to process (will be modified in-place)
     * @param region         Custom watermark region (position + size)
     * @param force_variant  Profile to source the alpha map from
     *                       (V2 default; pass V1 for legacy outputs)
     */
    void remove_watermark_custom(
        cv::Mat& image,
        const cv::Rect& region,
        std::optional<WatermarkVariant> force_variant = std::nullopt
    );

    /**
     * Add watermark to an image (Gemini-style)
     *
     * @param image     The image to process (will be modified in-place)
     * @param force_size Force a specific watermark size (auto-detect if nullopt)
     */
    void add_watermark(
        cv::Mat& image,
        std::optional<WatermarkSize> force_size = std::nullopt
    );

    /**
     * Add watermark at a custom region with interpolated alpha map
     *
     * @param image     The image to process (will be modified in-place)
     * @param region    Custom watermark region (position + size)
     */
    void add_watermark_custom(
        cv::Mat& image,
        const cv::Rect& region
    );

    /**
     * Get the alpha map for a specific size (for external use)
     */
    const cv::Mat& get_alpha_map(WatermarkSize size) const;

    /**
     * Apply inpaint cleanup on residual artifacts after reverse alpha blend.
     *
     * Uses SPARSE MASK derived from alpha map gradient — only repairs the
     * sparkle edge pixels where interpolation broke the math, leaving
     * correctly-restored pixels untouched.
     *
     * Two-stage pipeline:
     *   1. Reverse alpha blend removes ~90% of watermark (mathematical precision)
     *   2. This function cleans the remaining edge artifacts using cv::inpaint
     *
     * @param image       Image after reverse alpha blend (modified in-place)
     * @param region      The watermark region (where reverse alpha was applied)
     * @param strength    Blend strength: 0.0 = keep reverse-alpha result,
     *                                    1.0 = fully replace with inpainted result
     * @param method      Inpaint method (TELEA or NS)
     * @param inpaint_radius  Inpaint radius for cv::inpaint (default: 3)
     * @param padding     Context padding around region in pixels (default: 16)
     */
    void inpaint_residual(
        cv::Mat& image,
        const cv::Rect& region,
        float strength = 0.85f,
        InpaintMethod method = InpaintMethod::NS,
        int inpaint_radius = 10,
        int padding = 32
    ) const;

    /// Get a reference to the alpha map for a given size + variant.
    const cv::Mat& get_alpha_map(WatermarkSize size, WatermarkVariant variant) const;

private:
    // Per-variant alpha maps (CV_32FC1, 0.0 - 1.0).
    cv::Mat alpha_map_small_;       // V1 small (legacy alias used by older paths)
    cv::Mat alpha_map_large_;       // V1 large (legacy alias used by older paths)
    cv::Mat alpha_map_small_v2_;    // V2 small
    cv::Mat alpha_map_large_v2_;    // V2 large
    bool has_v2_{false};            // Whether V2 assets were loaded
    float logo_value_;              // Logo brightness (255 = white)

    cv::Mat& get_alpha_map_mutable(WatermarkSize size);

    /**
     * Create an interpolated alpha map for a custom size
     * Uses bilinear interpolation from the 96x96 alpha map
     */
    cv::Mat create_interpolated_alpha(int target_width, int target_height,
                                       WatermarkVariant variant) const;

    /**
     * Resolve the alpha map to use for (size, variant) on an image of the
     * given dimensions. For V2 small the canonical template is 36x36
     * (1024-class outputs); larger "small" outputs (e.g. 1376x768) carry a
     * proportionally scaled logo, so the template is interpolated from the
     * 96px large source to config.logo_size.
     */
    cv::Mat effective_alpha_map(WatermarkSize size, WatermarkVariant variant,
                                int image_width, int image_height) const;

    // Helpers to initialise alpha maps from cv::Mat captures.
    void init_alpha_maps(const cv::Mat& bg_small, const cv::Mat& bg_large);
    void init_alpha_maps_v2(const cv::Mat& bg_small, const cv::Mat& bg_large);

    // Run the three-stage detection at a single variant. Used internally
    // by detect_watermark() to try both variants and pick the higher.
    DetectionResult detect_one_variant(
        const cv::Mat& image,
        std::optional<WatermarkSize> force_size,
        WatermarkVariant variant
    ) const;
};

/**
 * Result of processing an image
 */
struct ProcessResult {
    bool success;              // Whether processing succeeded
    bool skipped;              // Whether processing was skipped (no watermark detected)
    float confidence;          // Detection confidence (if detection was used)
    std::string message;       // Status message
};

/**
 * Process a single image file
 *
 * @param input_path   Input image path
 * @param output_path  Output image path
 * @param remove       Remove watermark (true) or add watermark (false)
 * @param engine       The watermark engine to use
 * @param force_size   Force a specific watermark size (auto-detect if nullopt)
 * @param use_detection  Enable watermark detection before processing
 * @param detection_threshold  Confidence threshold for detection (default: 0.25)
 * @return             Processing result
 */
ProcessResult process_image(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    bool remove,
    WatermarkEngine& engine,
    std::optional<WatermarkSize> force_size = std::nullopt,
    bool use_detection = false,
    float detection_threshold = 0.25f,
    std::optional<WatermarkVariant> force_variant = std::nullopt
);

} // namespace gwt
