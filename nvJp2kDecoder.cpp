/*
 * Copyright (c) 2024 WangBin <wbsecg1 at gmail.com>
 * This file is part of MDK
 * MDK SDK: https://github.com/wang-bin/mdk-sdk
 * Free for opensource softwares or non-commercial use.
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * This software contains source code provided by NVIDIA Corporation.
 */
#include "mdk/VideoDecoder.h"
#include "mdk/MediaInfo.h"
#include "mdk/Packet.h"
#include "mdk/VideoFrame.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "base/log.h"

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

using namespace std;

#define NVJP2K_ENSURE(expr, ...) NVJP2K_RUN(expr, return __VA_ARGS__)
#define NVJP2K_WARN(expr, ...) NVJP2K_RUN(expr)
#define NVJP2K_RUN(EXPR, ...) do { \
        const auto nvjp2k_ret__ = (EXPR); \
        if (nvjp2k_ret__ != NVJPEG2K_STATUS_SUCCESS) { \
            std::clog << fmt::to_string("nvJPEG2000 ERROR@%d %s " #EXPR " : %d", __LINE__, __func__, nvjp2k_ret__) << std::endl << std::flush; \
            __VA_ARGS__; \
        } \
    } while (false)


#define CUDA_ENSURE(expr, ...) CUDA_RUN(expr, return __VA_ARGS__)
#define CUDA_WARN(expr, ...) CUDA_RUN(expr)
#define CUDA_RUN(EXPR, ...)  do { \
        const cudaError_t cuda_err_ = (EXPR);  \
        if (cuda_err_ != cudaSuccess) {  \
            std::clog << fmt::to_string("CUDA runtime ERROR@%d %s " #EXPR " : (%d) %s", __LINE__, __func__, cuda_err_, cudaGetErrorString(cuda_err_)) << std::endl << std::flush; \
            __VA_ARGS__; \
        }  \
    } while (false)


MDK_NS_BEGIN

class nvJp2kDecoder final : public VideoDecoder
{
public:
    const char* name() const override {return "nvjp2k";}
    bool open() override;
    bool close() override;
    bool flush() override;
    int decode(const Packet& pkt) override;
private:
    nvjpeg2kDecodeState_t state_ = nullptr;
    nvjpeg2kHandle_t handle_ = nullptr;
    cudaStream_t cu_stream_ = nullptr;
    nvjpeg2kStream_t stream_ = nullptr;
    nvjpeg2kDecodeParams_t params_ = nullptr;
    VideoFormat fmt_;
    nvjpeg2kImage_t img_ = {};
    uint16_t* data16_[4] = {};
    uint8_t* data8_[4] = {};
    size_t pitch_[4] = {};

    bool rgb_ = false;
    bool packed_ = false;
};

bool nvJp2kDecoder::open()
{
    int ver[3] = {};
    int cuver[3] = {};
    NVJP2K_ENSURE(nvjpeg2kGetProperty(libraryPropertyType::MAJOR_VERSION, &ver[0]), false);
    NVJP2K_ENSURE(nvjpeg2kGetProperty(libraryPropertyType::MINOR_VERSION, &ver[1]), false);
    NVJP2K_ENSURE(nvjpeg2kGetProperty(libraryPropertyType::PATCH_LEVEL, &ver[2]), false);
    NVJP2K_ENSURE(nvjpeg2kGetCudartProperty(libraryPropertyType::MAJOR_VERSION, &cuver[0]), false);
    NVJP2K_ENSURE(nvjpeg2kGetCudartProperty(libraryPropertyType::MINOR_VERSION, &cuver[1]), false);
    NVJP2K_ENSURE(nvjpeg2kGetCudartProperty(libraryPropertyType::PATCH_LEVEL, &cuver[2]), false);
    clog << fmt::to_string("nvjpeg2k version: %d.%d.%d.%d/%d.%d.%d, cudart: %d.%d.%d"
        , NVJPEG2K_VER_MAJOR, NVJPEG2K_VER_MINOR, NVJPEG2K_VER_PATCH, NVJPEG2K_VER_BUILD
        , ver[0], ver[1], ver[2], cuver[0], cuver[1], cuver[2]) << endl;

    fmt_ = parameters().format;
    const auto rgb = get_or("rgb", "0") == "1";
    const auto planar = stoi(get_or("planar", "-1")); // 0: packed; 1: planar; -1: auto, packed for rgb(because of uncommon planar rgb formats), planar otherwise
    packed_ = planar == 0 || (planar < 0 && fmt_.isRGB()) || rgb;
    if (packed_) // 420 422 input?
        packed_ = !fmt_.isPlanar() && fmt_.subsampleWidth(1) == 1 && fmt_.subsampleHeight(1) == 1; // required by nvjpeg2000
    rgb_ = packed_;
    if (rgb_ && !fmt_.isRGB()) {
        if (fmt_.bitsPerChannel() > 8)
            fmt_ = PixelFormat::RGB48LE; // FIXME: channel shift
        else
            fmt_ = PixelFormat::RGB24;
    }
    if (!packed_) {
        if (fmt_.isXYZ())
            fmt_ = PixelFormat::XYZ12PLE;
        if (fmt_.isRGB())
            fmt_ = PixelFormat::YUV444P; // FIXME:
    }

    CUDA_ENSURE(cudaStreamCreateWithFlags(&cu_stream_, cudaStreamNonBlocking), false);
    NVJP2K_ENSURE(nvjpeg2kCreateSimple(&handle_), false);
    NVJP2K_ENSURE(nvjpeg2kDecodeStateCreate(handle_, &state_), false);
    NVJP2K_ENSURE(nvjpeg2kStreamCreate(&stream_), false);
    NVJP2K_ENSURE(nvjpeg2kDecodeParamsCreate(&params_), false);
    NVJP2K_ENSURE(nvjpeg2kDecodeParamsSetRGBOutput(params_, rgb_ || packed_), false);
    NVJP2K_ENSURE(nvjpeg2kDecodeParamsSetOutputFormat(params_, packed_ ? NVJPEG2K_FORMAT_INTERLEAVED : NVJPEG2K_FORMAT_PLANAR), false);

    onOpen();
    return true;
}

bool nvJp2kDecoder::close()
{
    if (img_.pixel_data) {
        for (uint32_t i = 0; i < fmt_.planeCount(); i++) {
            CUDA_WARN(cudaFree(img_.pixel_data[i]));
        }
    }
    NVJP2K_WARN(nvjpeg2kDecodeParamsDestroy(params_));
    NVJP2K_WARN(nvjpeg2kStreamDestroy(stream_));
    NVJP2K_WARN(nvjpeg2kDestroy(handle_));
    CUDA_WARN(cudaStreamDestroy(cu_stream_));
    onClose();
    return true;
}

bool nvJp2kDecoder::flush()
{
    onFlush();
    return true;
}

int nvJp2kDecoder::decode(const Packet& pkt)
{
    if (pkt.isEnd())
        return INT_MAX;

    NVJP2K_ENSURE(nvjpeg2kStreamParse(handle_, pkt.buffer->constData(), pkt.buffer->size(), 0, 0, stream_), -1);
    nvjpeg2kImageInfo_t info;
    NVJP2K_ENSURE(nvjpeg2kStreamGetImageInfo(stream_, &info), -1);
    vector<nvjpeg2kImageComponentInfo_t> comp_info(info.num_components);
    for (uint32_t c = 0; c < info.num_components; ++c) {
        NVJP2K_ENSURE(nvjpeg2kStreamGetImageComponentInfo(stream_, &comp_info[c], c), -1);
    }

    if (!img_.pixel_data) { // assume same size
        img_.num_components = info.num_components;
        img_.pitch_in_bytes = pitch_;
        if (comp_info[0].precision > 8 && comp_info[0].precision < 16) {
            img_.pixel_data = (void**)data16_;
            img_.pixel_type = comp_info[0].sgn ? NVJPEG2K_INT16 : NVJPEG2K_UINT16;
        } else if (comp_info[0].precision == 8) {
            img_.pixel_data = (void**)data8_;
            img_.pixel_type = NVJPEG2K_UINT8;
        } else {
            clog << fmt::to_string("nvjp2k unsupported precision: %d", comp_info[0].precision) << endl;
            return -1;
        }
        if (packed_) { // ?
            CUDA_ENSURE(cudaMallocPitch(&img_.pixel_data[0], &img_.pitch_in_bytes[0], info.image_width * fmt_.bytesPerPixel(), info.image_height), -1);
        } else {
            for(uint32_t c = 0; c < info.num_components; c++) {
                CUDA_ENSURE(cudaMallocPitch(&img_.pixel_data[c], &img_.pitch_in_bytes[c], comp_info[c].component_width * fmt_.bytesPerPixel(c), comp_info[c].component_height), -1);
            }
        }
    }

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_WARN(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CUDA_WARN(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));
    CUDA_WARN(cudaEventRecord(start, cu_stream_));
    NVJP2K_ENSURE(nvjpeg2kDecodeImage(handle_, state_, stream_, params_, &img_, cu_stream_), -1);
    CUDA_WARN(cudaEventRecord(stop, cu_stream_));
    CUDA_WARN(cudaEventSynchronize(stop));
    float elapsed = 0; // us
    CUDA_WARN(cudaEventElapsedTime(&elapsed, start, stop));

    CUDA_WARN(cudaEventDestroy(start));
    CUDA_WARN(cudaEventDestroy(stop));

    VideoFrame frame(info.image_width, info.image_height, fmt_);
    frame.setBuffers(nullptr);
    for (int i = 0; i < fmt_.planeCount(); ++i) {
        auto data = img_.pixel_data[i];
        CUDA_ENSURE(cudaMemcpy2DAsync(frame.buffer(i)->data(), frame.buffer(i)->stride(), data, img_.pitch_in_bytes[i], comp_info[i].component_width * fmt_.bytesPerPixel(i), comp_info[i].component_height, cudaMemcpyDeviceToHost, cu_stream_), -1);
    }
    CUDA_WARN(cudaStreamSynchronize(cu_stream_));
    frame.setTimestamp(pkt.pts);
    frameDecoded(std::move(frame));
    return 0;
}

static void register_video_decoders_nvjp2k() {
    VideoDecoder::registerOnce("nvjp2k", []{return new nvJp2kDecoder();});
}
MDK_NS_END

// project name must be nvjp2k or mdk-nvjp2k
MDK_PLUGIN(nvjp2k) {
    using namespace MDK_NS;
    register_video_decoders_nvjp2k();
    return MDK_ABI_VERSION;
}