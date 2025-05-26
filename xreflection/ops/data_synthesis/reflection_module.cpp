#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <chrono>
#include <iomanip>  

namespace py = pybind11;
 
float truncatedNormal(float mean, float stddev, float minVal, float maxVal, std::default_random_engine& generator) {
    std::normal_distribution<float> distribution(mean, stddev);
    float number;
    do {
        number = distribution(generator);
    } while (number < minVal || number > maxVal);
    return number;
}

cv::Mat convolve2D(const cv::Mat& input, const cv::Mat& kernel, int borderType = cv::BORDER_DEFAULT) {
    cv::Mat output;
    cv::filter2D(input, output, input.depth(), kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    return output;
}

py::tuple full_reflection_synthesis(py::array_t<float> T_np, py::array_t<float> R_np) {
    auto T_buf = T_np.request(), R_buf = R_np.request();

    if (T_buf.ndim != 3 || T_buf.shape[2] != 3) {
        throw std::runtime_error("Input array T must have shape (h, w, 3)");
    }

    if (R_buf.ndim != 3 || R_buf.shape[2] != 3) {
        throw std::runtime_error("Input array R must have shape (h, w, 3)");
    }
    int h = T_buf.shape[0], w = T_buf.shape[1];

    cv::Mat T(h, w, CV_32FC3, T_buf.ptr);
    cv::Mat R(h, w, CV_32FC3, R_buf.ptr);

    T /= 255.0f;
    R /= 255.0f;

    int kernel_sizes[] = {5, 7, 9, 11};
    std::vector<float> probs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::discrete_distribution<> dist_kernel(probs.begin(), probs.end()); 
     
    std::uniform_real_distribution<> dist_sigma(2, 5);
    std::uniform_real_distribution<> dist_beta(0.8, 1.0);
    std::uniform_real_distribution<> dist_merge(0.0, 1.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);

    int kernel_size = kernel_sizes[dist_kernel(rng)];
    float sigma = dist_sigma(rng);
    float beta = dist_beta(rng);

    cv::Mat kernel = cv::getGaussianKernel(kernel_size, sigma, CV_32F);
    cv::Mat kernel2d = kernel * kernel.t();
    cv::Mat separableBlurredR;

    kernel2d.convertTo(kernel2d, CV_32F);
    separableBlurredR = convolve2D(R, kernel2d, cv::BORDER_DEFAULT);
    float a1_loc = 1.109f, a1_scale = 0.118f;
    float a1_min = (0.82f - a1_loc) / a1_scale, a1_max = (1.42f - a1_loc) / a1_scale;
    float a2_loc = 1.106f, a2_scale = 0.115f;
    float a2_min = (0.85f - a2_loc) / a2_scale, a2_max = (1.35f - a2_loc) / a2_scale;
    float a3_loc = 1.078f, a3_scale = 0.116f;
    float a3_min = (0.85f - a3_loc) / a3_scale, a3_max = (1.31f - a3_loc) / a3_scale;

    std::vector<cv::Mat> channels(3);
    cv::split(T, channels);
    float channel1_rand = truncatedNormal(a1_loc, a1_scale, a1_min, a1_max, rng);

    float channel2_rand = truncatedNormal(a2_loc, a2_scale, a2_min, a2_max, rng);

    float channel3_rand = truncatedNormal(a3_loc, a3_scale, a3_min, a3_max, rng);

    channels[0] *= channel1_rand;
    channels[1] *= channel2_rand;
    channels[2] *= channel3_rand;

    cv::merge(channels, T);
    cv::Mat I, R_mod = separableBlurredR * beta;
    if (dist_merge(rng) < 0.7f) {
    I = T + R_mod - T.mul(R_mod);
    } else {
    I = T + R_mod;
    }

    double maxVal;
    cv::minMaxLoc(I, nullptr, &maxVal);

    if (maxVal > 1.0) {
        std::vector<float> overValues;
        for (int i = 0; i < I.rows; ++i) {
            for (int j = 0; j < I.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    float pixelValue = I.at<cv::Vec3f>(i, j)[c];
                    if (pixelValue > 1.0f) {
                        overValues.push_back(pixelValue);
                    }
                }
            }
        }

        float m = 0.0f;
        if (!overValues.empty()) {
            m = std::accumulate(overValues.begin(), overValues.end(), 0.0f) / overValues.size();
            m = (m - 1.0f) * 1.3f;
        }
        cv::Mat m_mat(R_mod.size(), R_mod.type(), cv::Scalar(m, m, m));

        cv::Mat R_minus_m = R_mod - m_mat;
         

        cv::Mat R_clipped = R_minus_m.clone();
        for (int i = 0; i < R_clipped.rows; ++i) {
            for (int j = 0; j < R_clipped.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    float& pixelValue = R_clipped.at<cv::Vec3f>(i, j)[c];
                    pixelValue = cv::min(cv::max(pixelValue, 0.0f), 1.0f);
                }
            }
        }
    I = T + R_clipped;
    cv::Mat I_clipped = I.clone();
    for (int i = 0; i < I_clipped.rows; ++i) {
        for (int j = 0; j < I_clipped.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                float& pixelValue = I_clipped.at<cv::Vec3f>(i, j)[c];
                pixelValue = cv::min(cv::max(pixelValue, 0.0f), 1.0f);
            }
        }
    }
    I = I_clipped;
}

    py::array_t<float> T_out({h, w, 3});
    py::array_t<float> R_out({h, w, 3});
    py::array_t<float> I_out({h, w, 3});

    std::memcpy(T_out.mutable_data(), T.data, sizeof(float) * h * w * 3);
    std::memcpy(R_out.mutable_data(), R_mod.data, sizeof(float) * h * w * 3);
    std::memcpy(I_out.mutable_data(), I.data, sizeof(float) * h * w * 3);
     
     
    return py::make_tuple(T_out, R_out, I_out);
}

PYBIND11_MODULE(reflection_module, m) {
    m.def("full_reflection_synthesis", &full_reflection_synthesis, "C++ fast version of reflection synthesis");
}