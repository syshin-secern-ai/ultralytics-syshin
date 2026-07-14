// yunet26n rknn_raw 포즈 모델 추론기 — RKNNBackend._decode_stride의 C++ 이식.
// 입력: BGR cv::Mat, 출력: 원본 이미지 좌표계의 얼굴 박스 + 5 키포인트.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "rknn_api.h"

struct Det {
    cv::Rect2f box;  // xywh, 원본 이미지 좌표
    float score;
    int cls;        // 클래스 id (단일 클래스 모델은 항상 0)
    float kpt[15];  // 5 x (x, y, conf), 원본 이미지 좌표
};

class RknnPose {
public:
    static constexpr int imgsz = 640;

    bool init(const char* model_path) {
        if (rknn_init(&ctx, (void*)model_path, 0, 0, nullptr) < 0) return false;
        rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        oattr.resize(io_num.n_output);
        for (uint32_t i = 0; i < io_num.n_output; i++) {
            oattr[i].index = i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oattr[i], sizeof(rknn_tensor_attr));
            printf("out[%u] %s dims=[%u,%u,%u,%u]\n", i, oattr[i].name,
                   oattr[i].dims[0], oattr[i].dims[1], oattr[i].dims[2], oattr[i].dims[3]);
        }
        return true;
    }

    std::vector<Det> infer(const cv::Mat& img, float conf_thres, float iou_thres) {
        // 전처리: letterbox(114) + BGR->RGB, uint8 NHWC (1/255 정규화는 .rknn에 내장됨)
        float gain = std::min((float)imgsz / img.cols, (float)imgsz / img.rows);
        int nw = lround(img.cols * gain), nh = lround(img.rows * gain);
        int pad_x = (imgsz - nw) / 2, pad_y = (imgsz - nh) / 2;
        cv::Mat resized, input(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::resize(img, resized, {nw, nh});
        resized.copyTo(input(cv::Rect(pad_x, pad_y, nw, nh)));
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

        rknn_input in{};
        in.index = 0;
        in.type = RKNN_TENSOR_UINT8;
        in.fmt = RKNN_TENSOR_NHWC;
        in.size = imgsz * imgsz * 3;
        in.buf = input.data;
        rknn_inputs_set(ctx, 1, &in);
        rknn_run(ctx, nullptr);

        // want_float=1 -> 런타임이 int8을 float32로 역양자화해서 반환
        std::vector<rknn_output> outs(io_num.n_output);
        for (uint32_t i = 0; i < io_num.n_output; i++) {
            memset(&outs[i], 0, sizeof(rknn_output));
            outs[i].index = i;
            outs[i].want_float = 1;
        }
        rknn_outputs_get(ctx, io_num.n_output, outs.data(), nullptr);

        // 디코딩. 출력 순서: bbox_s8, cls_s8, bbox_s16, cls_s16, bbox_s32, cls_s32, kpt_s8, kpt_s16, kpt_s32
        // bbox=(1,4,H,W) raw ltrb 그리드 단위, cls=(1,1,H,W) sigmoid 완료, kpt=(1,15,H,W) 앵커 오프셋+conf
        std::vector<Det> dets;
        const int nl = 3;
        for (int l = 0; l < nl; l++) {
            const float* bbox = (const float*)outs[2 * l].buf;
            const float* cls = (const float*)outs[2 * l + 1].buf;
            const float* kpt = (const float*)outs[2 * nl + l].buf;
            const int nc = oattr[2 * l + 1].dims[1];  // 클래스 수 (cls 텐서 채널)
            const int H = oattr[2 * l].dims[2], W = oattr[2 * l].dims[3];
            const float s = (float)imgsz / W;
            const int hw = H * W;
            for (int i = 0; i < hw; i++) {
                // 앵커당 최고 점수 클래스 하나만 사용 (non_max_suppression의 multi_label=False와 동일)
                int best = 0;
                for (int c = 1; c < nc; c++)
                    if (cls[c * hw + i] > cls[best * hw + i]) best = c;
                float score = cls[best * hw + i];
                if (score < conf_thres) continue;
                float ax = i % W + 0.5f, ay = i / W + 0.5f;
                float lt = bbox[i], tp = bbox[hw + i], rt = bbox[2 * hw + i], bt = bbox[3 * hw + i];
                Det d;
                d.score = score;
                d.cls = best;
                float cx = (ax + (rt - lt) * 0.5f) * s, cy = (ay + (bt - tp) * 0.5f) * s;
                float w = (lt + rt) * s, h = (tp + bt) * s;
                d.box = {cx - w / 2, cy - h / 2, w, h};
                for (int k = 0; k < 5; k++) {
                    d.kpt[3 * k] = (kpt[(3 * k) * hw + i] + ax) * s;
                    d.kpt[3 * k + 1] = (kpt[(3 * k + 1) * hw + i] + ay) * s;
                    d.kpt[3 * k + 2] = kpt[(3 * k + 2) * hw + i];
                }
                dets.push_back(d);
            }
        }
        rknn_outputs_release(ctx, io_num.n_output, outs.data());

        // 클래스별 NMS — non_max_suppression의 기본 동작(agnostic=False)과 동일.
        // 클래스마다 박스를 크게 offset시켜 NMSBoxes 한 번으로 처리 (NMSBoxesBatched는 OpenCV 4.7+ 전용)
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        for (const auto& d : dets) {
            float off = d.cls * 7680.f;
            boxes.emplace_back(d.box.x + off, d.box.y + off, d.box.width, d.box.height);
            scores.push_back(d.score);
        }
        std::vector<int> idx;
        cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, idx);
        std::vector<Det> keep;
        for (int i : idx) keep.push_back(dets[i]);

        // letterbox 역변환 -> 원본 이미지 좌표 (scale_boxes/scale_coords처럼 이미지 경계로 clip)
        const float iw = (float)img.cols, ih = (float)img.rows;
        for (auto& d : keep) {
            float x1 = std::clamp((d.box.x - pad_x) / gain, 0.f, iw);
            float y1 = std::clamp((d.box.y - pad_y) / gain, 0.f, ih);
            float x2 = std::clamp((d.box.x + d.box.width - pad_x) / gain, 0.f, iw);
            float y2 = std::clamp((d.box.y + d.box.height - pad_y) / gain, 0.f, ih);
            d.box = {x1, y1, x2 - x1, y2 - y1};
            for (int k = 0; k < 5; k++) {
                d.kpt[3 * k] = std::clamp((d.kpt[3 * k] - pad_x) / gain, 0.f, iw);
                d.kpt[3 * k + 1] = std::clamp((d.kpt[3 * k + 1] - pad_y) / gain, 0.f, ih);
            }
        }
        return keep;
    }

    ~RknnPose() {
        if (ctx) rknn_destroy(ctx);
    }

private:
    rknn_context ctx = 0;
    rknn_input_output_num io_num{};
    std::vector<rknn_tensor_attr> oattr;
};

// 검출 결과를 이미지에 그린다 (박스 초록, 키포인트 빨강 + conf 텍스트).
inline void draw_dets(cv::Mat& img, const std::vector<Det>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(img, d.box, {0, 255, 0}, 2);
        char score_txt[8];
        snprintf(score_txt, sizeof(score_txt), "%.2f", d.score);
        cv::putText(img, score_txt, {(int)d.box.x, std::max((int)d.box.y - 4, 12)}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {0, 255, 0}, 1, cv::LINE_AA);
        for (int k = 0; k < 5; k++) {
            float x = d.kpt[3 * k], y = d.kpt[3 * k + 1], c = d.kpt[3 * k + 2];
            if (c < 0.25f) continue;
            cv::circle(img, {(int)x, (int)y}, 2, {0, 0, 255}, -1);
            char txt[8];
            snprintf(txt, sizeof(txt), "%.2f", c);
            cv::putText(img, txt, {(int)x + 3, (int)y - 3}, cv::FONT_HERSHEY_SIMPLEX, 0.36, {255, 255, 255}, 1,
                        cv::LINE_AA);
        }
    }
}
