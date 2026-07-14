// 카메라 스트림 추론 (predict_rknn.py의 C++ 버전):
//   MJPEG 수신(http://host:cam_port/) -> 추론 -> 뷰어 전송(tcp host:viewer_port, 4바이트 BE 길이 + JPEG)
//   ./pose_rknn_stream model.rknn [host=172.16.151.180] [cam_port=5000] [viewer_port=5001] [conf=0.25] [iou=0.7]
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include "rknn_pose.hpp"

// send()는 부분 전송될 수 있으므로 전부 보낼 때까지 반복한다.
static bool send_all(int sock, const void* data, size_t len) {
    const char* p = (const char*)data;
    while (len > 0) {
        ssize_t n = send(sock, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s model.rknn [host] [cam_port] [viewer_port] [conf] [iou]\n", argv[0]);
        return 1;
    }
    const std::string host = argc > 2 ? argv[2] : "172.16.151.180";
    const int cam_port = argc > 3 ? atoi(argv[3]) : 5000;
    const int viewer_port = argc > 4 ? atoi(argv[4]) : 5001;
    const float conf = argc > 5 ? atof(argv[5]) : 0.25f;
    const float iou = argc > 6 ? atof(argv[6]) : 0.7f;

    RknnPose model;
    if (!model.init(argv[1])) {
        printf("rknn_init failed\n");
        return 1;
    }

    cv::VideoCapture cap("http://" + host + ":" + std::to_string(cam_port) + "/");
    if (!cap.isOpened()) {
        printf("BOARD: Failed to open camera stream.\n");
        return 1;
    }

    // 추론이 카메라 fps보다 느리면 VideoCapture 버퍼에 오래된 프레임이 쌓여 지연이 누적되므로,
    // 읽기 스레드가 프레임을 계속 소비하며 최신 한 장만 남기고 나머지는 버린다.
    std::mutex mtx;
    cv::Mat latest;
    std::atomic<bool> running{true};
    std::thread reader([&] {
        cv::Mat f;
        while (running) {
            if (!cap.read(f)) {
                running = false;
                break;
            }
            std::lock_guard<std::mutex> lk(mtx);
            f.copyTo(latest);
        }
    });

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    // WSL 미러드 네트워킹에서는 뷰어가 닫혀도 FIN/RST가 안 오는 경우가 있어 send가 버퍼만 채우다
    // 무기한 블록된다. 송신 타임아웃을 걸어 그 경우에도 send 실패로 종료되게 한다.
    timeval tmo{5, 0};
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tmo, sizeof(tmo));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(viewer_port);
    hostent* he = gethostbyname(host.c_str());
    if (he) memcpy(&addr.sin_addr, he->h_addr, he->h_length);
    if (!he || connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("BOARD: Failed to connect to viewer at %s:%d\n", host.c_str(), viewer_port);
        running = false;
        reader.join();
        return 1;
    }
    printf("BOARD: Connected to viewer at %s:%d\n", host.c_str(), viewer_port);

    cv::Mat img;
    std::vector<uchar> buf;
    while (true) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (!latest.empty()) {
                latest.copyTo(img);
                latest.release();
            } else {
                img.release();
            }
        }
        if (img.empty()) {
            if (!running) {
                printf("BOARD: No frame.\n");
                break;
            }
            usleep(5000);
            continue;
        }

        draw_dets(img, model.infer(img, conf, iou));

        if (!cv::imencode(".jpg", img, buf)) continue;
        uint32_t len = htonl((uint32_t)buf.size());
        if (!send_all(sock, &len, 4) || !send_all(sock, buf.data(), buf.size())) {
            printf("BOARD: Viewer closed (q pressed). Exiting.\n");
            break;
        }
    }

    running = false;
    close(sock);
    cap.release();
    reader.join();
    return 0;
}
