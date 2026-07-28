/**
 * @file    watermark_engine.cpp
 * @brief   Gemini Watermark Tool - Watermark Engine
 * @author  AllenK (Kwyshell)
 * @date    2025.12.13
 * @license MIT
 *
 * @details
 * Watermark Engine Implementation
 *
 * @see https://github.com/allenk/GeminiWatermarkTool
 */

#include "core/watermark_engine.hpp"
#include "core/blend_modes.hpp"
#include "utils/image_io.hpp"
#include "utils/path_formatter.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <stdexcept>
#include <atomic>
#include <chrono>

namespace gwt {

namespace {

// Resolve V2 small position by inferring the canonical large source the
// image was downscaled from. Small Gemini outputs are 1024-class on the
// long side and inherit per-axis rounding from the source aspect ratio,
// so a single fixed margin does not fit every aspect.
WatermarkPosition v2_small_config_from_dims(int W, int H) {
    const int long_side  = std::max(W, H);
    const int short_side = std::min(W, H);

    // Map the output back to the canonical large width.
    // Half-scale outputs (free-tier 1376/1408/1424-class, issue #40) identify
    // their canonical directly: twice the long side lands on 2752/2816/2848.
    // 1024-class outputs are inferred from the short side; thresholds bisect
    // the observed canonical heights (540, 559, 572).
    double source_long_dim;
    if (long_side > 1100) {
        const double doubled = 2.0 * long_side;
        source_long_dim = 2752.0;
        for (double cand : {2816.0, 2848.0}) {
            if (std::abs(doubled - cand) < std::abs(doubled - source_long_dim)) {
                source_long_dim = cand;
            }
        }
    } else if (short_side >= 566) {
        source_long_dim = 2752.0;
    } else if (short_side >= 550) {
        source_long_dim = 2816.0;
    } else {
        source_long_dim = 2848.0;
    }

    const double scale = static_cast<double>(long_side) / source_long_dim;
    const int margin = static_cast<int>(std::round(192.0 * scale));
    // Logo size scales with the canonical source like the margin does.
    // 1024-class outputs land at ~35-36 px where the canonical 36 template
    // is known to match either rounding, so keep the validated 36 there.
    // Larger "small" outputs (e.g. free-tier 1376x768 = exactly half of
    // canonical 2752x1536) carry a proportionally bigger logo (48 px) with
    // V2 opacity -- see issue #40.
    const int ideal = static_cast<int>(std::round(96.0 * scale));
    return WatermarkPosition{
        .margin_right = margin,
        .margin_bottom = margin,
        .logo_size = (ideal <= 40) ? 36 : ideal
    };
}

}  // namespace

WatermarkPosition get_watermark_config(int image_width, int image_height,
                                       WatermarkVariant variant) {
    const bool is_large = (image_width > 1024 && image_height > 1024);
    if (variant == WatermarkVariant::V1) {
        // Legacy profile
        if (is_large) {
            return WatermarkPosition{
                .margin_right = 64,
                .margin_bottom = 64,
                .logo_size = 96
            };
        }
        return WatermarkPosition{
            .margin_right = 32,
            .margin_bottom = 32,
            .logo_size = 48
        };
    }
    // V2 (current profile). Large stays at 192 margin / 96 logo, which
    // matches the canonical 2752/2816/2848-wide outputs exactly. Small
    // outputs need aspect-aware scaling -- see v2_small_config_from_dims().
    if (is_large) {
        return WatermarkPosition{
            .margin_right = 192,
            .margin_bottom = 192,
            .logo_size = 96
        };
    }
    return v2_small_config_from_dims(image_width, image_height);
}

WatermarkSize get_watermark_size(int image_width, int image_height) {
    // Large (96x96) only when BOTH dimensions >= 1024
    // 1024x1024 is Small
    if (image_width > 1024 && image_height > 1024) {
        return WatermarkSize::Large;
    }
    return WatermarkSize::Small;
}

void WatermarkEngine::init_alpha_maps_v2(const cv::Mat& bg_small, const cv::Mat& bg_large) {
    cv::Mat small_resized = bg_small;
    cv::Mat large_resized = bg_large;

    // Resize if needed -- V2 canonical sizes are 36x36 (small) and 96x96 (large)
    if (small_resized.cols != 36 || small_resized.rows != 36) {
        cv::resize(small_resized, small_resized, cv::Size(36, 36), 0, 0, cv::INTER_AREA);
    }
    if (large_resized.cols != 96 || large_resized.rows != 96) {
        cv::resize(large_resized, large_resized, cv::Size(96, 96), 0, 0, cv::INTER_AREA);
    }

    alpha_map_small_v2_ = calculate_alpha_map(small_resized);
    alpha_map_large_v2_ = calculate_alpha_map(large_resized);
    has_v2_ = true;
}

// Helper function to initialize alpha maps
void WatermarkEngine::init_alpha_maps(const cv::Mat& bg_small, const cv::Mat& bg_large) {
    cv::Mat small_resized = bg_small;
    cv::Mat large_resized = bg_large;

    // Resize if needed
    if (small_resized.cols != 48 || small_resized.rows != 48) {
        spdlog::warn("Small capture is {}x{}, expected 48x48. Resizing.",
                     small_resized.cols, small_resized.rows);
        cv::resize(small_resized, small_resized, cv::Size(48, 48), 0, 0, cv::INTER_AREA);
    }

    if (large_resized.cols != 96 || large_resized.rows != 96) {
        spdlog::warn("Large capture is {}x{}, expected 96x96. Resizing.",
                     large_resized.cols, large_resized.rows);
        cv::resize(large_resized, large_resized, cv::Size(96, 96), 0, 0, cv::INTER_AREA);
    }

    // Calculate alpha maps from background
    // alpha = bg_value / 255
    alpha_map_small_ = calculate_alpha_map(small_resized);
    alpha_map_large_ = calculate_alpha_map(large_resized);

    spdlog::debug("Alpha map small: {}x{}, large: {}x{}",
                  alpha_map_small_.cols, alpha_map_small_.rows,
                  alpha_map_large_.cols, alpha_map_large_.rows);

    // Log alpha statistics for debugging
    double min_val, max_val;
    cv::minMaxLoc(alpha_map_large_, &min_val, &max_val);
    spdlog::debug("Large alpha map range: {:.4f} - {:.4f}", min_val, max_val);
}

WatermarkEngine::WatermarkEngine(
    const std::filesystem::path& bg_small,
    const std::filesystem::path& bg_large,
    float logo_value)
    : logo_value_(logo_value) {

    // Load background captures from files
    cv::Mat bg_small_bk = cv::imread(bg_small.string(), cv::IMREAD_COLOR);
    if (bg_small_bk.empty()) {
        throw std::runtime_error("Failed to load small background capture: " + bg_small.string());
    }

    cv::Mat bg_large_bk = cv::imread(bg_large.string(), cv::IMREAD_COLOR);
    if (bg_large_bk.empty()) {
        throw std::runtime_error("Failed to load large background capture: " + bg_large.string());
    }

    init_alpha_maps(bg_small_bk, bg_large_bk);
    spdlog::info("Loaded background captures from files");
}

// Six-argument constructor: V1 + V2 assets together.
WatermarkEngine::WatermarkEngine(
    const unsigned char* v1_small, size_t v1_small_size,
    const unsigned char* v1_large, size_t v1_large_size,
    const unsigned char* v2_small, size_t v2_small_size,
    const unsigned char* v2_large, size_t v2_large_size,
    float logo_value)
    : logo_value_(logo_value) {

    auto decode = [](const unsigned char* data, size_t size, const char* label) {
        std::vector<unsigned char> buf(data, data + size);
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error(std::string("Failed to decode embedded ") + label);
        }
        return img;
    };

    cv::Mat bg_v1_small = decode(v1_small, v1_small_size, "V1 small BG capture");
    cv::Mat bg_v1_large = decode(v1_large, v1_large_size, "V1 large BG capture");
    cv::Mat bg_v2_small = decode(v2_small, v2_small_size, "V2 small BG capture");
    cv::Mat bg_v2_large = decode(v2_large, v2_large_size, "V2 large BG capture");

    init_alpha_maps(bg_v1_small, bg_v1_large);
    init_alpha_maps_v2(bg_v2_small, bg_v2_large);
    spdlog::info("Loaded embedded BG captures (V1 + V2 profiles)");
}

// Four-argument constructor (legacy). Loads V1 only; V2 detection is disabled.
WatermarkEngine::WatermarkEngine(
    const unsigned char* png_data_small, size_t png_size_small,
    const unsigned char* png_data_large, size_t png_size_large,
    float logo_value)
    : logo_value_(logo_value) {

    // Decode PNG from memory
    std::vector<unsigned char> buf_small(png_data_small, png_data_small + png_size_small);
    std::vector<unsigned char> buf_large(png_data_large, png_data_large + png_size_large);

    cv::Mat bg_small = cv::imdecode(buf_small, cv::IMREAD_COLOR);
    if (bg_small.empty()) {
        throw std::runtime_error("Failed to decode embedded small background capture");
    }

    cv::Mat bg_large = cv::imdecode(buf_large, cv::IMREAD_COLOR);
    if (bg_large.empty()) {
        throw std::runtime_error("Failed to decode embedded large background capture");
    }

    init_alpha_maps(bg_small, bg_large);
    spdlog::info("Loaded embedded background captures (standalone mode)");
}

void WatermarkEngine::remove_watermark(
    cv::Mat& image,
    std::optional<WatermarkSize> force_size,
    std::optional<WatermarkVariant> force_variant) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    // Ensure BGR format
    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    // Default to V2 (current profile). Callers passing force_variant
    // pin the choice. If the engine was built without V2 assets fall back
    // to V1 unconditionally.
    const WatermarkVariant variant = force_variant.value_or(
        has_v2_ ? WatermarkVariant::V2 : WatermarkVariant::V1);

    // Determine watermark size
    const WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows)
    );

    // Run detection so the V2 small snap refinement applies here too --
    // formula position rounds inconsistently across source aspect ratios,
    // and we want remove to operate on the same exact pixels the detector
    // matched.
    const DetectionResult det = detect_one_variant(image, size, variant);
    const cv::Point pos(det.region.x, det.region.y);
    const cv::Mat alpha_map = effective_alpha_map(size, variant, image.cols, image.rows);

    spdlog::debug("Removing watermark at ({}, {}) with {}x{} alpha map (size: {}, variant: {})",
                  pos.x, pos.y, alpha_map.cols, alpha_map.rows,
                  size == WatermarkSize::Small ? "Small" : "Large",
                  variant == WatermarkVariant::V1 ? "V1" : "V2");

    // Apply reverse alpha blending
    remove_watermark_alpha_blend(image, alpha_map, pos, logo_value_);
}


void WatermarkEngine::add_watermark(
    cv::Mat& image,
    std::optional<WatermarkSize> force_size) {
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    // Ensure BGR format
    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    // Determine watermark size
    WatermarkSize size = force_size.value_or(
        get_watermark_size(image.cols, image.rows)
    );

    // Get position config based on actual size used
    WatermarkPosition config;
    if (size == WatermarkSize::Small) {
        config = WatermarkPosition{32, 32, 48};
    } else {
        config = WatermarkPosition{64, 64, 96};
    }

    cv::Point pos = config.get_position(image.cols, image.rows);
    const cv::Mat& alpha_map = get_alpha_map(size);

    spdlog::debug("Adding watermark at ({}, {}) with {}x{} alpha map (size: {})",
                  pos.x, pos.y, alpha_map.cols, alpha_map.rows,
                  size == WatermarkSize::Small ? "Small" : "Large");

    // Apply alpha blending
    add_watermark_alpha_blend(image, alpha_map, pos, logo_value_);
}

cv::Mat& WatermarkEngine::get_alpha_map_mutable(WatermarkSize size) {
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

const cv::Mat& WatermarkEngine::get_alpha_map(WatermarkSize size) const {
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

const cv::Mat& WatermarkEngine::get_alpha_map(WatermarkSize size,
                                              WatermarkVariant variant) const {
    if (variant == WatermarkVariant::V2 && has_v2_) {
        return (size == WatermarkSize::Small) ? alpha_map_small_v2_ : alpha_map_large_v2_;
    }
    return (size == WatermarkSize::Small) ? alpha_map_small_ : alpha_map_large_;
}

// =============================================================================
// Watermark Detection (Three-Stage Algorithm)
// =============================================================================

DetectionResult WatermarkEngine::detect_watermark(
    const cv::Mat& image,
    std::optional<WatermarkSize> force_size,
    std::optional<WatermarkVariant> force_variant) const
{
    // Default to V2 (current profile). Callers that need V1 (legacy Gemini
    // outputs) pass force_variant explicitly -- the CLI --legacy flag and
    // the GUI Legacy checkbox both wire through here.
    const WatermarkVariant variant = force_variant.value_or(WatermarkVariant::V2);
    return detect_one_variant(image, force_size, variant);
}

DetectionResult WatermarkEngine::detect_one_variant(
    const cv::Mat& image,
    std::optional<WatermarkSize> force_size,
    WatermarkVariant variant) const
{
    DetectionResult result{};
    result.detected = false;
    result.confidence = 0.0f;
    result.spatial_score = 0.0f;
    result.gradient_score = 0.0f;
    result.variance_score = 0.0f;
    result.variant = variant;

    if (image.empty()) {
        return result;
    }

    // Determine watermark size and position for the requested variant
    const WatermarkSize size = force_size.value_or(get_watermark_size(image.cols, image.rows));
    const WatermarkPosition config = get_watermark_config(image.cols, image.rows, variant);
    cv::Point pos = config.get_position(image.cols, image.rows);
    const cv::Mat alpha_map = effective_alpha_map(size, variant, image.cols, image.rows);

    // V2 small position is derived from the inferred canonical source
    // dimensions and accurate to ~1-3 px in most cases. A narrow NCC sweep
    // absorbs the residual rounding noise without giving content artefacts
    // a chance to outscore the actual watermark on busy backgrounds.
    const bool needs_snap = (variant == WatermarkVariant::V2 && size == WatermarkSize::Small);
    const int snap_pad = needs_snap ? 3 : 0;

    result.size = size;
    result.region = cv::Rect(pos.x, pos.y, alpha_map.cols, alpha_map.rows);

    // Calculate ROI (clamp to image bounds), expanded by snap_pad for V2 small.
    const int x1 = std::max(0, pos.x - snap_pad);
    const int y1 = std::max(0, pos.y - snap_pad);
    const int x2 = std::min(image.cols, pos.x + alpha_map.cols + snap_pad);
    const int y2 = std::min(image.rows, pos.y + alpha_map.rows + snap_pad);

    if (x1 >= x2 || y1 >= y2) {
        spdlog::debug("Detection: ROI out of bounds");
        return result;
    }

    // Extract region and convert to grayscale
    const cv::Rect image_roi(x1, y1, x2 - x1, y2 - y1);
    const cv::Mat region = image(image_roi);
    cv::Mat gray_region;

    if (region.channels() >= 3) {
        cv::cvtColor(region, gray_region, cv::COLOR_BGR2GRAY);
    } else {
        gray_region = region.clone();
    }

    // Convert to float [0, 1]
    cv::Mat gray_f;
    gray_region.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    // For snap mode the alpha region is the full alpha map (NCC slides it
    // across the wider gray region); for non-snap it's clamped to the
    // image bounds in case the formula position pokes outside.
    cv::Mat alpha_region;
    if (needs_snap) {
        alpha_region = alpha_map;
    } else {
        const cv::Rect alpha_roi(x1 - pos.x, y1 - pos.y, x2 - x1, y2 - y1);
        alpha_region = alpha_map(alpha_roi);
    }

    // =========================================================================
    // Stage 1: Spatial Structural Correlation (NCC)
    // The watermark's diamond/star pattern should correlate with the alpha map
    // =========================================================================
    cv::Mat spatial_match;
    cv::matchTemplate(gray_f, alpha_region, spatial_match, cv::TM_CCOEFF_NORMED);

    double min_spatial, spatial_score;
    cv::Point match_loc;
    cv::minMaxLoc(spatial_match, &min_spatial, &spatial_score, nullptr, &match_loc);
    result.spatial_score = static_cast<float>(spatial_score);

    // If we snapped, the best NCC offset gives us the refined absolute
    // position -- but only trust it when the correlation is strong enough.
    // Busy backgrounds (low NCC) can pull the match a few pixels toward
    // content artefacts; falling back to the geometry-derived formula
    // position is more reliable there.
    if (needs_snap && spatial_score >= 0.60) {
        pos.x = x1 + match_loc.x;
        pos.y = y1 + match_loc.y;
        result.region = cv::Rect(pos.x, pos.y, alpha_map.cols, alpha_map.rows);
    }

    // Circuit Breaker: If spatial correlation is too low, definitely no watermark
    constexpr double kSpatialThreshold = 0.25;
    if (spatial_score < kSpatialThreshold) {
        spdlog::debug("Detection: spatial={:.3f} < {:.2f}, rejected",
                      spatial_score, kSpatialThreshold);
        result.confidence = static_cast<float>(spatial_score * 0.5);  // Return low confidence
        return result;
    }

    // =========================================================================
    // Stage 2: Gradient-Domain Correlation (Edge Signature)
    // Watermark edges should match alpha map edges
    // =========================================================================
    cv::Mat img_gx, img_gy, img_gmag;
    cv::Sobel(gray_f, img_gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray_f, img_gy, CV_32F, 0, 1, 3);
    cv::magnitude(img_gx, img_gy, img_gmag);

    cv::Mat alpha_gx, alpha_gy, alpha_gmag;
    cv::Sobel(alpha_region, alpha_gx, CV_32F, 1, 0, 3);
    cv::Sobel(alpha_region, alpha_gy, CV_32F, 0, 1, 3);
    cv::magnitude(alpha_gx, alpha_gy, alpha_gmag);

    cv::Mat grad_match;
    cv::matchTemplate(img_gmag, alpha_gmag, grad_match, cv::TM_CCOEFF_NORMED);

    double min_grad, grad_score;
    cv::minMaxLoc(grad_match, &min_grad, &grad_score);
    result.gradient_score = static_cast<float>(grad_score);

    // =========================================================================
    // Stage 3: Statistical Variance Analysis (Texture Dampening)
    // Watermarks reduce texture variance in the affected region
    // =========================================================================
    double var_score = 0.0;
    const int ref_h = std::min(y1, config.logo_size);

    if (ref_h > 8) {
        // Use region above watermark as reference
        const cv::Rect ref_roi(x1, y1 - ref_h, x2 - x1, ref_h);
        const cv::Mat ref_region = image(ref_roi);
        cv::Mat gray_ref;

        if (ref_region.channels() >= 3) {
            cv::cvtColor(ref_region, gray_ref, cv::COLOR_BGR2GRAY);
        } else {
            gray_ref = ref_region;
        }

        cv::Scalar m_wm, s_wm, m_ref, s_ref;
        cv::meanStdDev(gray_region, m_wm, s_wm);
        cv::meanStdDev(gray_ref, m_ref, s_ref);

        if (s_ref[0] > 5.0) {
            // Watermarks dampen high-frequency background variance
            var_score = std::clamp(1.0 - (s_wm[0] / s_ref[0]), 0.0, 1.0);
        }
    }
    result.variance_score = static_cast<float>(var_score);

    // =========================================================================
    // Heuristic Fusion: Weighted Ensemble
    // =========================================================================
    const double confidence =
        (spatial_score * 0.50) +   // Spatial correlation is most important
        (grad_score * 0.30) +      // Edge signature
        (var_score * 0.20);        // Variance dampening

    result.confidence = static_cast<float>(std::clamp(confidence, 0.0, 1.0));

    // Position-anchored spatial rescue: on busy backgrounds the gradient and
    // variance stages collapse toward 0 and drag the fused confidence below
    // the gate even when the spatial NCC -- anchored at the formula position
    // (+-3px for V2 small) -- is unambiguous. Accept a strong anchored match
    // on its own. Calibrated against removed/clean corners: highest observed
    // negative was 0.26 (near-uniform background), positives 0.30+.
    constexpr double kSpatialRescue = 0.30;
    if (spatial_score >= kSpatialRescue) {
        result.confidence = std::max(result.confidence,
                                     static_cast<float>(spatial_score));
    }

    // Determine if watermark is detected based on confidence threshold
    constexpr float kDetectionThreshold = 0.35f;
    result.detected = (result.confidence >= kDetectionThreshold);

    spdlog::debug("Detection: spatial={:.3f}, grad={:.3f}, var={:.3f} -> conf={:.3f} ({})",
                  spatial_score, grad_score, var_score, result.confidence,
                  result.detected ? "DETECTED" : "not detected");

    return result;
}

cv::Mat WatermarkEngine::effective_alpha_map(WatermarkSize size, WatermarkVariant variant,
                                             int image_width, int image_height) const {
    const cv::Mat& base = get_alpha_map(size, variant);
    // V2 small: the canonical 36x36 template only fits 1024-class outputs.
    // Newer, larger "small" outputs carry a proportionally scaled logo
    // (config.logo_size), so interpolate from the 96px large source.
    if (size == WatermarkSize::Small && variant == WatermarkVariant::V2 && has_v2_) {
        const WatermarkPosition config =
            get_watermark_config(image_width, image_height, variant);
        if (config.logo_size != base.cols) {
            return create_interpolated_alpha(config.logo_size, config.logo_size, variant);
        }
    }
    return base;
}

cv::Mat WatermarkEngine::create_interpolated_alpha(int target_width, int target_height,
                                                    WatermarkVariant variant) const {
    // Use the 96x96 large alpha map of the active profile as source
    // (higher resolution = better quality on resize).
    const cv::Mat& source = get_alpha_map(WatermarkSize::Large, variant);

    if (target_width == source.cols && target_height == source.rows) {
        return source.clone();
    }

    cv::Mat interpolated;

    // Use INTER_LINEAR (bilinear) for upscaling, INTER_AREA for downscaling
    int interp_method = (target_width > source.cols || target_height > source.rows)
                        ? cv::INTER_LINEAR
                        : cv::INTER_AREA;

    cv::resize(source, interpolated, cv::Size(target_width, target_height), 0, 0, interp_method);

    spdlog::debug("Created interpolated alpha map: {}x{} -> {}x{} (method: {})",
                  source.cols, source.rows, target_width, target_height,
                  interp_method == cv::INTER_LINEAR ? "bilinear" : "area");

    return interpolated;
}

void WatermarkEngine::remove_watermark_custom(
    cv::Mat& image,
    const cv::Rect& region,
    std::optional<WatermarkVariant> force_variant)
{
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    // Ensure BGR format
    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    // Source the alpha map from the active profile (V2 by default).
    const WatermarkVariant variant = force_variant.value_or(
        has_v2_ ? WatermarkVariant::V2 : WatermarkVariant::V1);

    const cv::Point pos(region.x, region.y);

    // Fast paths when the user-selected region matches a canonical size.
    if (region.width == 36 && region.height == 36) {
        spdlog::info("Custom region matches 36x36, using current-profile small alpha");
        remove_watermark_alpha_blend(image, get_alpha_map(WatermarkSize::Small, variant),
                                     pos, logo_value_);
        return;
    }
    // 48x48 is only canonical for the legacy (V1) profile. Newer free-tier
    // outputs (1376-class) carry a 48px logo with V2 opacity, so the legacy
    // fast path applies only when the variant actually resolves to V1;
    // V2 falls through to the interpolated 96->48 map (issue #40).
    if (region.width == 48 && region.height == 48 && variant == WatermarkVariant::V1) {
        spdlog::info("Custom region matches 48x48, using legacy-profile small alpha");
        remove_watermark_alpha_blend(image, get_alpha_map(WatermarkSize::Small, WatermarkVariant::V1),
                                     pos, logo_value_);
        return;
    }
    if (region.width == 96 && region.height == 96) {
        spdlog::info("Custom region matches 96x96, using large alpha map (profile {})",
                     variant == WatermarkVariant::V1 ? "V1" : "V2");
        remove_watermark_alpha_blend(image, get_alpha_map(WatermarkSize::Large, variant),
                                     pos, logo_value_);
        return;
    }

    // Arbitrary size: resize the active profile's large alpha map.
    cv::Mat custom_alpha = create_interpolated_alpha(region.width, region.height, variant);

    spdlog::info("Removing watermark at ({},{}) with custom {}x{} alpha map (profile {})",
                 pos.x, pos.y, region.width, region.height,
                 variant == WatermarkVariant::V1 ? "V1" : "V2");

    remove_watermark_alpha_blend(image, custom_alpha, pos, logo_value_);
}

void WatermarkEngine::add_watermark_custom(
    cv::Mat& image,
    const cv::Rect& region)
{
    if (image.empty()) {
        throw std::runtime_error("Empty image provided");
    }

    // Ensure BGR format
    if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    // Check for exact match with standard sizes
    if (region.width == 48 && region.height == 48) {
        cv::Point pos(region.x, region.y);
        add_watermark_alpha_blend(image, alpha_map_small_, pos, logo_value_);
        return;
    }

    if (region.width == 96 && region.height == 96) {
        cv::Point pos(region.x, region.y);
        add_watermark_alpha_blend(image, alpha_map_large_, pos, logo_value_);
        return;
    }

    // Create interpolated alpha map for custom size (add path uses the
    // legacy profile -- new watermarks are added with the original alpha)
    cv::Mat custom_alpha = create_interpolated_alpha(region.width, region.height,
                                                     WatermarkVariant::V1);
    cv::Point pos(region.x, region.y);

    spdlog::info("Adding watermark at ({},{}) with custom {}x{} alpha map",
                 pos.x, pos.y, region.width, region.height);

    add_watermark_alpha_blend(image, custom_alpha, pos, logo_value_);
}

// =============================================================================
// Guided Multi-Scale Detection (Snap Engine)
// =============================================================================

GuidedDetectionResult WatermarkEngine::guided_detect(
    const cv::Mat& image,
    const cv::Rect& search_rect,
    std::atomic<bool>* cancel_flag,
    int min_size,
    int max_size) const
{
    auto start_time = std::chrono::high_resolution_clock::now();

    GuidedDetectionResult result{};

    if (image.empty() || search_rect.width < 8 || search_rect.height < 8) {
        return result;
    }

    // Clamp search rect to image bounds
    cv::Rect search = search_rect & cv::Rect(0, 0, image.cols, image.rows);
    if (search.width < 8 || search.height < 8) {
        return result;
    }

    // Clamp min/max to sensible range
    min_size = std::max(min_size, 16);
    max_size = std::min(max_size, std::min(search.width, search.height));
    if (min_size > max_size) {
        spdlog::debug("guided_detect: min_size {} > max_size {}, no search possible",
                      min_size, max_size);
        return result;
    }

    // =========================================================================
    // Size preference factor
    //
    // NCC has an inherent bias toward smaller templates: a 24x24 patch can
    // trivially find a high-correlation match within any watermark region.
    // To counter this, we weight scores by a size factor that prefers
    // templates closer to the known standard size (96px).
    //
    //   adjusted_score = raw_ncc * cbrt(scale / kReferenceSize)
    //
    // Using cbrt (cube root) instead of sqrt gives a gentler penalty for
    // small watermarks (~28px preview tier) while still preferring larger
    // matches when NCC scores are similar.
    //
    //   scale  28: weight = 0.66  (sqrt was 0.54)
    //   scale  32: weight = 0.69  (sqrt was 0.58)
    //   scale  48: weight = 0.79  (sqrt was 0.71)
    //   scale  96: weight = 1.00  (same)
    // =========================================================================
    constexpr double kReferenceSize = 96.0;

    auto size_adjusted_score = [&](double raw_ncc, int scale) -> double {
        double weight = std::cbrt(static_cast<double>(scale) / kReferenceSize);
        weight = std::min(weight, 1.0);  // Cap at 1.0 for scales >= 96
        return raw_ncc * weight;
    };

    // Extract grayscale search region
    cv::Mat search_region = image(search);
    cv::Mat gray_region;
    if (search_region.channels() >= 3) {
        cv::cvtColor(search_region, gray_region, cv::COLOR_BGR2GRAY);
    } else {
        gray_region = search_region.clone();
    }
    cv::Mat gray_f;
    gray_region.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    // Use alpha_map_large_ (96x96) as template source for best quality
    const cv::Mat& source_alpha = alpha_map_large_;

    // =========================================================================
    // Phase 1: Coarse search — large steps for quick scan
    // =========================================================================
    struct Candidate {
        cv::Point position;       // Position within search region
        int scale;                // Template size
        double raw_score;         // Raw NCC score
        double adjusted_score;    // Size-adjusted score
    };

    constexpr int kCoarseScaleStep = 4;
    constexpr int kTopK = 5;

    std::vector<Candidate> coarse_candidates;
    coarse_candidates.reserve(kTopK);

    // Generate scale list
    std::vector<int> coarse_scales;
    for (int s = min_size; s <= max_size; s += kCoarseScaleStep) {
        coarse_scales.push_back(s);
    }
    // Always include the exact standard sizes if in range
    for (int std_size : {48, 96}) {
        if (std_size >= min_size && std_size <= max_size) {
            bool already_present = false;
            for (int s : coarse_scales) {
                if (std::abs(s - std_size) <= 2) { already_present = true; break; }
            }
            if (!already_present) coarse_scales.push_back(std_size);
        }
    }
    std::sort(coarse_scales.begin(), coarse_scales.end());

    result.total_scales = static_cast<int>(coarse_scales.size());

    spdlog::debug("guided_detect: searching {} scales [{}-{}] in {}x{} region",
                  coarse_scales.size(), min_size, max_size,
                  search.width, search.height);

    for (int scale : coarse_scales) {
        // Check cancellation
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
            result.was_cancelled = true;
            spdlog::debug("guided_detect: cancelled at scale {}", scale);
            break;
        }

        // Template must fit within search region
        if (scale > gray_f.cols || scale > gray_f.rows) {
            result.scales_searched++;
            continue;
        }

        // Resize alpha template to current scale
        cv::Mat tmpl;
        cv::resize(source_alpha, tmpl, cv::Size(scale, scale), 0, 0,
                   scale > source_alpha.cols ? cv::INTER_LINEAR : cv::INTER_AREA);

        // NCC template matching
        cv::Mat match_result;
        cv::matchTemplate(gray_f, tmpl, match_result, cv::TM_CCOEFF_NORMED);

        // Find best match at this scale
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(match_result, &min_val, &max_val, &min_loc, &max_loc);

        result.scales_searched++;

        double adj = size_adjusted_score(max_val, scale);

        spdlog::debug("  scale {:3d}: raw_ncc={:.3f} adjusted={:.3f}", scale, max_val, adj);

        if (adj > 0.08) {  // Minimal threshold for coarse phase (on adjusted score)
            Candidate c{max_loc, scale, max_val, adj};

            // Insert into sorted top-K by adjusted score
            if (static_cast<int>(coarse_candidates.size()) < kTopK) {
                coarse_candidates.push_back(c);
                std::sort(coarse_candidates.begin(), coarse_candidates.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.adjusted_score > b.adjusted_score;
                          });
            } else if (adj > coarse_candidates.back().adjusted_score) {
                coarse_candidates.back() = c;
                std::sort(coarse_candidates.begin(), coarse_candidates.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.adjusted_score > b.adjusted_score;
                          });
            }
        }
    }

    if (coarse_candidates.empty()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        spdlog::info("guided_detect: no candidates found in {} us ({} scales)",
                     elapsed, result.scales_searched);
        return result;
    }

    spdlog::debug("guided_detect: top {} coarse candidates:", coarse_candidates.size());
    for (const auto& c : coarse_candidates) {
        spdlog::debug("  scale={} pos=({},{}) raw={:.3f} adj={:.3f}",
                      c.scale, c.position.x, c.position.y, c.raw_score, c.adjusted_score);
    }

    // =========================================================================
    // Phase 2: Fine refinement around top candidates
    // =========================================================================
    constexpr int kFineScaleStep = 2;
    constexpr int kFineScaleRange = 10;  // +/- 10 pixels around candidate scale

    Candidate best{cv::Point(0, 0), 0, -1.0, -1.0};

    for (const auto& candidate : coarse_candidates) {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
            result.was_cancelled = true;
            break;
        }

        // Refine scale around candidate
        int scale_lo = std::max(min_size, candidate.scale - kFineScaleRange);
        int scale_hi = std::min(max_size, candidate.scale + kFineScaleRange);

        for (int s = scale_lo; s <= scale_hi; s += kFineScaleStep) {
            if (s > gray_f.cols || s > gray_f.rows) continue;

            cv::Mat tmpl;
            cv::resize(source_alpha, tmpl, cv::Size(s, s), 0, 0,
                       s > source_alpha.cols ? cv::INTER_LINEAR : cv::INTER_AREA);

            cv::Mat match_result;
            cv::matchTemplate(gray_f, tmpl, match_result, cv::TM_CCOEFF_NORMED);

            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(match_result, &min_val, &max_val, &min_loc, &max_loc);

            double adj = size_adjusted_score(max_val, s);
            if (adj > best.adjusted_score) {
                best = Candidate{max_loc, s, max_val, adj};
            }
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();

    if (best.adjusted_score > 0.08) {
        // Convert position from search-region-relative to image-absolute
        result.found = true;
        result.confidence = static_cast<float>(best.adjusted_score);
        result.raw_ncc = static_cast<float>(best.raw_score);
        result.match_rect = cv::Rect(
            search.x + best.position.x,
            search.y + best.position.y,
            best.scale, best.scale
        );
        result.detected_size = best.scale;

        spdlog::info("guided_detect: found at ({},{}) size {}x{} "
                     "raw_ncc={:.3f} adjusted={:.3f} "
                     "in {} us ({} coarse scales, {} candidates refined)",
                     result.match_rect.x, result.match_rect.y,
                     best.scale, best.scale,
                     best.raw_score, best.adjusted_score,
                     elapsed, result.scales_searched,
                     coarse_candidates.size());
    } else {
        spdlog::info("guided_detect: no match above threshold in {} us", elapsed);
    }

    return result;
}

// =============================================================================
// Inpaint Residual Cleanup
// =============================================================================

void WatermarkEngine::inpaint_residual(
    cv::Mat& image,
    const cv::Rect& region,
    float strength,
    InpaintMethod method,
    int inpaint_radius,
    int padding) const
{
    if (image.empty() || region.width < 4 || region.height < 4) {
        return;
    }

    strength = std::clamp(strength, 0.0f, 1.0f);
    if (strength < 0.001f) {
        return;
    }

    // =========================================================================
    // Common: Calculate padded region
    // =========================================================================
    cv::Rect padded_rect(
        region.x - padding,
        region.y - padding,
        region.width + padding * 2,
        region.height + padding * 2
    );
    padded_rect &= cv::Rect(0, 0, image.cols, image.rows);

    if (padded_rect.width < 8 || padded_rect.height < 8) {
        spdlog::warn("inpaint_residual: padded region too small");
        return;
    }

    cv::Rect inner_rect(
        region.x - padded_rect.x,
        region.y - padded_rect.y,
        region.width,
        region.height
    );
    inner_rect &= cv::Rect(0, 0, padded_rect.width, padded_rect.height);

    const char* method_name =
        (method == InpaintMethod::GAUSSIAN) ? "Soft Inpaint" :
        (method == InpaintMethod::TELEA)    ? "TELEA" : "NS";

    // =========================================================================
    // GAUSSIAN (Soft Inpaint): gradient weight + boundary feather
    // =========================================================================
    if (method == InpaintMethod::GAUSSIAN) {
        // --- gradient weight (proven effective, unchanged) ---

        // Compute alpha gradient to locate sparkle edges
        const cv::Mat& source_alpha = alpha_map_large_;
        cv::Mat alpha_resized;
        int interp = (region.width > source_alpha.cols)
            ? cv::INTER_LINEAR : cv::INTER_AREA;
        cv::resize(source_alpha, alpha_resized,
                   cv::Size(region.width, region.height), 0, 0, interp);

        cv::Mat grad_x, grad_y, grad_mag;
        cv::Sobel(alpha_resized, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(alpha_resized, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, grad_mag);

        double grad_min, grad_max;
        cv::minMaxLoc(grad_mag, &grad_min, &grad_max);
        if (grad_max <= grad_min) {
            spdlog::info("inpaint_residual: flat gradient, no edges found");
            return;
        }

        // Normalize to 0.0-1.0
        cv::Mat grad_norm = (grad_mag - grad_min) / (grad_max - grad_min);

        // Gamma correction: sqrt expands weak gradient values
        cv::Mat grad_weight;
        cv::sqrt(grad_norm, grad_weight);

        // Dilate to cover residual spread (v5 original: 5×5)
        cv::Mat dk = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(grad_weight, grad_weight, dk);

        // Smooth weight for natural transitions (v5 original: σ=2.0)
        cv::GaussianBlur(grad_weight, grad_weight, cv::Size(0, 0), 2.0);

        // Scale by user strength
        grad_weight *= strength;
        cv::threshold(grad_weight, grad_weight, 1.0, 1.0, cv::THRESH_TRUNC);

        // --- Smooth boundary transition (replaces v5 hard cutoff) ---
        // Embed gradient weight into padded coordinate system
        cv::Mat weight = cv::Mat::zeros(padded_rect.size(), CV_32F);
        grad_weight.copyTo(weight(inner_rect));

        // Tiny blur to soften the boundary transition
        // σ=1.0 affects only ~3px at the inner_rect edge
        // Interior (21px from edge on 42×42) is effectively unchanged
        // Tips at boundary: weight transitions smoothly instead of hard cutoff
        cv::GaussianBlur(weight, weight, cv::Size(0, 0), 1.0);

        // --- Gaussian blur the image ---
        int ksize = inpaint_radius * 2 + 1;
        if (ksize % 2 == 0) ksize++;
        ksize = std::max(ksize, 3);
        double sigma = inpaint_radius * 0.8;

        cv::Mat padded_area = image(padded_rect).clone();
        cv::Mat blurred;
        cv::GaussianBlur(padded_area, blurred, cv::Size(ksize, ksize), sigma);

        // --- Per-pixel weighted blend ---
        cv::Mat dst = image(padded_rect);
        cv::Mat weight_3ch;
        cv::merge(std::vector<cv::Mat>{weight, weight, weight}, weight_3ch);

        cv::Mat dst_f, blurred_f, result_f;
        dst.convertTo(dst_f, CV_32FC3);
        blurred.convertTo(blurred_f, CV_32FC3);

        cv::Mat one_minus_w = cv::Scalar(1.0, 1.0, 1.0) - weight_3ch;
        cv::multiply(dst_f, one_minus_w, dst_f);
        cv::multiply(blurred_f, weight_3ch, blurred_f);
        result_f = dst_f + blurred_f;

        result_f.convertTo(dst, CV_8UC3);

        int active_pixels = cv::countNonZero(weight > 0.01f);
        spdlog::info("inpaint_residual: {}, strength={:.0f}%, radius={}, sigma={:.1f}, "
                     "{} active pixels",
                     method_name, strength * 100.0f, inpaint_radius, sigma,
                     active_pixels);
        return;
    }

    // =========================================================================
    // TELEA / NS: Sparse gradient mask + cv::inpaint
    //
    // Uses alpha map gradient to identify sparkle edges, creates binary mask,
    // then runs OpenCV inpaint for structural reconstruction.
    // These methods are best for edge artifacts on high-contrast boundaries
    // (e.g. carpet ↔ floor edges that Gaussian blur would smear).
    // =========================================================================

    // Compute alpha gradient to locate sparkle edges
    const cv::Mat& source_alpha = alpha_map_large_;
    cv::Mat alpha_resized;
    int interp = (region.width > source_alpha.cols) ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(source_alpha, alpha_resized,
               cv::Size(region.width, region.height), 0, 0, interp);

    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(alpha_resized, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(alpha_resized, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    // Normalize to 0-255
    double grad_min, grad_max;
    cv::minMaxLoc(grad_mag, &grad_min, &grad_max);
    if (grad_max <= grad_min) {
        spdlog::info("inpaint_residual: flat gradient, no edges found");
        return;
    }

    cv::Mat grad_u8;
    grad_mag.convertTo(grad_u8, CV_8U,
                       255.0 / (grad_max - grad_min),
                       -grad_min * 255.0 / (grad_max - grad_min));

    // Create sparse binary mask at sparkle edges
    cv::Mat sparse_mask;
    cv::threshold(grad_u8, sparse_mask, 20, 255, cv::THRESH_BINARY);

    // Dilate to cover residual spread (1-2px beyond gradient peak)
    cv::Mat dilate_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(sparse_mask, sparse_mask, dilate_kernel);

    int masked_pixels = cv::countNonZero(sparse_mask);
    if (masked_pixels == 0) {
        spdlog::info("inpaint_residual: no edge pixels found, skipping");
        return;
    }

    spdlog::info("inpaint_residual: {} sparse mask {}/{} pixels ({:.1f}%), "
                 "strength={:.0f}%",
                 method_name, masked_pixels, region.width * region.height,
                 100.0f * masked_pixels / (region.width * region.height),
                 strength * 100.0f);

    // Embed mask into padded coordinate system
    cv::Mat mask = cv::Mat::zeros(padded_rect.size(), CV_8UC1);
    sparse_mask.copyTo(mask(inner_rect));

    // Run cv::inpaint
    cv::Mat padded_area = image(padded_rect).clone();
    int cv_method = (method == InpaintMethod::TELEA)
        ? cv::INPAINT_TELEA
        : cv::INPAINT_NS;

    cv::Mat inpainted;
    cv::inpaint(padded_area, mask, inpainted, inpaint_radius, cv_method);

    // Blend at masked pixels only (unmasked pixels stay untouched)
    cv::Mat dst = image(padded_rect);
    cv::Mat src_inner = dst(inner_rect);
    cv::Mat inp_inner = inpainted(inner_rect);
    cv::Mat mask_inner = mask(inner_rect);

    if (strength >= 0.999f) {
        inp_inner.copyTo(src_inner, mask_inner);
    } else {
        cv::Mat blended;
        cv::addWeighted(src_inner, 1.0 - strength, inp_inner, strength, 0.0, blended);
        blended.copyTo(src_inner, mask_inner);
    }

    spdlog::info("inpaint_residual: applied {} at {:.0f}% strength, radius={}, "
                 "{} pixels repaired",
                 method_name, strength * 100.0f, inpaint_radius, masked_pixels);
}

ProcessResult process_image(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    bool remove,
    WatermarkEngine& engine,
    std::optional<WatermarkSize> force_size,
    bool use_detection,
    float detection_threshold,
    std::optional<WatermarkVariant> force_variant) {

    ProcessResult result{};
    result.success = false;
    result.skipped = false;
    result.confidence = 0.0f;

    try {
        // Read image
        cv::Mat image = cv::imread(input_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            result.message = "Failed to load image";
            spdlog::error("Failed to load image: {}", input_path);
            return result;
        }

        spdlog::info("Processing: {} ({}x{})",
                     input_path.filename(),
                     image.cols, image.rows);

        // Watermark detection (only for removal mode). The confidence
        // threshold is the safety gate: below it we skip, so that users
        // who accidentally drop a non-watermarked file into batch/simple
        // mode do not have their image silently altered.
        if (use_detection && remove) {
            DetectionResult detection = engine.detect_watermark(image, force_size, force_variant);
            result.confidence = detection.confidence;

            const bool ncc_passes = detection.detected ||
                                    detection.confidence >= detection_threshold;
            if (!ncc_passes) {
                result.skipped = true;
                result.success = true;  // Not an error, just skipped
                result.message = fmt::format("No watermark detected ({:.0f}%), skipped",
                                             detection.confidence * 100.0f);
                spdlog::info("{}: {} (spatial={:.2f}, grad={:.2f}, var={:.2f})",
                             input_path.filename(), result.message,
                             detection.spatial_score, detection.gradient_score,
                             detection.variance_score);
                return result;
            }

            spdlog::info("Watermark detected ({:.0f}% confidence), processing...",
                         detection.confidence * 100.0f);
        }

        // Process image
        if (remove) {
            engine.remove_watermark(image, force_size, force_variant);
        } else {
            engine.add_watermark(image, force_size);
        }

        // Create output directory if needed
        auto output_dir = output_path.parent_path();
        if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        // Determine output format and quality
        std::vector<int> params;
        std::string ext = output_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg") {
            params = {cv::IMWRITE_JPEG_QUALITY, 100};
        } else if (ext == ".png") {
            params = {cv::IMWRITE_PNG_COMPRESSION, 6};
        } else if (ext == ".webp") {
            params = {cv::IMWRITE_WEBP_QUALITY, 101};
        }

        // Write output. PNG goes through write_png() so non-ASCII paths
        // work on older Windows where cv::imwrite()'s path argument is
        // routed through the system ACP.
        bool write_success = (ext == ".png")
            ? write_png(output_path, image, 6)
            : cv::imwrite(output_path.string(), image, params);
        if (!write_success) {
            result.message = "Failed to write image";
            spdlog::error("Failed to write image: {}", output_path);
            return result;
        }

        result.success = true;
        result.message = remove ? "Watermark removed" : "Watermark added";
        spdlog::info("Saved: {}", output_path.filename());
        return result;

    } catch (const std::exception& e) {
        result.message = std::string("Error: ") + e.what();
        spdlog::error("Error processing {}: {}", input_path, e.what());
        return result;
    }
}

} // namespace gwt
