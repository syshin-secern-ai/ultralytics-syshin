// 단일 이미지 추론:
//   ./pose_rknn model.rknn image.jpg [conf=0.25] [iou=0.7]
// 결과를 표준 출력에 찍고 out.jpg로 저장한다.
#include "rknn_pose.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s model.rknn image.jpg [conf=0.25] [iou=0.7]\n", argv[0]);
        return 1;
    }
    const float conf = argc > 3 ? atof(argv[3]) : 0.25f;
    const float iou = argc > 4 ? atof(argv[4]) : 0.7f;

    RknnPose model;
    if (!model.init(argv[1])) {
        printf("rknn_init failed\n");
        return 1;
    }

    cv::Mat img = cv::imread(argv[2]);
    if (img.empty()) {
        printf("failed to read image: %s\n", argv[2]);
        return 1;
    }

    std::vector<Det> dets = model.infer(img, conf, iou);
    for (const auto& d : dets)
        printf("cls=%d %.3f  box=[%.0f, %.0f, %.0f, %.0f]\n", d.cls, d.score, d.box.x, d.box.y, d.box.width,
               d.box.height);

    draw_dets(img, dets);
    cv::imwrite("out.jpg", img);
    printf("%zu detections -> out.jpg\n", dets.size());
    return 0;
}
