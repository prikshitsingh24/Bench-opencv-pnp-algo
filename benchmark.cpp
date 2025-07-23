#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

// Camera intrinsics (same for all tests)
static const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    800, 0, 512,
    0, 800, 512,
    0, 0, 1);

static const cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

// ---------- Case 1 ----------
static void BM_SolvePnP_EPNP_Case1(benchmark::State& state) {
    std::vector<cv::Point3f> objectPoints = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };

    std::vector<cv::Point2f> imagePoints = {
        {500.0f, 500.0f},
        {600.0f, 500.0f},
        {600.0f, 600.0f},
        {500.0f, 600.0f}
    };

    cv::Mat rvec, tvec;
    for (auto _ : state) {
        bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(rvec);
        benchmark::DoNotOptimize(tvec);
    }
}

// ---------- Case 2 ----------
static void BM_SolvePnP_EPNP_Case2(benchmark::State& state) {
    std::vector<cv::Point3f> objectPoints = {
        {0.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f},
        {2.0f, 2.0f, 0.0f},
        {0.0f, 2.0f, 0.0f}
    };

    std::vector<cv::Point2f> imagePoints = {
        {400.0f, 400.0f},
        {700.0f, 400.0f},
        {700.0f, 700.0f},
        {400.0f, 700.0f}
    };

    cv::Mat rvec, tvec;
    for (auto _ : state) {
        bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(rvec);
        benchmark::DoNotOptimize(tvec);
    }
}

// ---------- Case 3 ----------
static void BM_SolvePnP_EPNP_Case3(benchmark::State& state) {
    std::vector<cv::Point3f> objectPoints = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 2.0f, 0.0f},
        {2.0f, 1.0f, 0.0f},
        {3.0f, 3.0f, 0.0f}
    };

    std::vector<cv::Point2f> imagePoints = {
        {520.0f, 480.0f},
        {580.0f, 500.0f},
        {590.0f, 550.0f},
        {610.0f, 600.0f}
    };

    cv::Mat rvec, tvec;
    for (auto _ : state) {
        bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
        benchmark::DoNotOptimize(success);
        benchmark::DoNotOptimize(rvec);
        benchmark::DoNotOptimize(tvec);
    }
}

// Register benchmarks
BENCHMARK(BM_SolvePnP_EPNP_Case1)->Iterations(1000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SolvePnP_EPNP_Case2)->Iterations(1000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SolvePnP_EPNP_Case3)->Iterations(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
