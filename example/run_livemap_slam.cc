#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

/** Split a string into a vector of strings separated by @a token.
 *
 * "this,is,a,string," -> ["this", "is", "a", "string", ""]
 */
inline std::vector<std::string> split(const std::string& str, const char token)
{
    std::vector<std::string> vec;
    size_t pos = 0, pos2 = 0;
    while ((pos2 = str.find(token, pos)) != std::string::npos) {
        vec.push_back(str.substr(pos, pos2 - pos));
        pos = pos2 + 1;
    }

    vec.push_back(str.substr(pos, str.length() - pos));
    return vec;
}

/// An image frame
struct frame {
    // path to color image, if color is empty
    std::string color_path;

    // loaded from data
    unsigned number = 0;
    double ts = 0; // sec
};

typedef std::vector<frame> frame_array;

class dataset {
public:
    /** Create dataset with path to a directory */
    dataset(const std::string& path) : m_path(path) {}

    /** Loads the dataset and returns true if successful */
    bool load();

    /** The frames loaded from the dataset */
    const frame_array& get_frames() const { return m_frames; }

    unsigned int start_frame = 0;
    unsigned int end_frame = std::numeric_limits<unsigned int>::max();

protected:
    std::string m_path;
    std::vector<frame> m_frames;
};

bool dataset::load()
{
    // expect a folder containing frame_XXXX.jpeg and a log.csv file
    std::ifstream is;
    const std::string logpath = m_path + "/log.csv";
    is.open(logpath);

    // skip header
    std::string line;
    std::getline(is, line);
    if (!is.good()) {
        spdlog::error("Could not load telemetry");
        return false;
    }

    unsigned last_frame = 0;
    while (std::getline(is, line)) {
        auto vec = split(line, ',');

        unsigned frame_number = m_frames.size() - 1;
        if (!vec[0].empty()) {
            frame_number = std::stod(vec[0]);
        }

        // if we dropped telemetry add a frame anyway
        if ( !m_frames.empty() && frame_number - last_frame > 1 ) {
            for (unsigned i = 0; i < frame_number - last_frame; ++i) {
                m_frames.push_back(m_frames.back());
                frame& f = m_frames.back();

                f.number = last_frame + i + 1;
                f.color_path = m_path + "/frame_" + std::to_string(f.number) + ".jpeg";
            }
        }

        m_frames.emplace_back();
        frame& f = m_frames.back();
        if (vec[7].empty()) {
            spdlog::error("Missing frame timestamp");
        }
        f.ts = std::stod(vec[7]);
        f.number = frame_number;
        last_frame = frame_number;
        f.color_path = m_path + "/frame_" + std::to_string(f.number) + ".jpeg";
    }

    return !m_frames.empty();
}

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const dataset& d, const std::string& mask_img_path,
                   const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
                   const bool eval_log, const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    const frame_array &frames = d.get_frames();

    std::vector<double> track_times;
    track_times.reserve(frames.size());

    // run the SLAM in another thread
    std::thread thread([&]() {

        double m_last_ts = -1;
        const float min_frame_gap = 0.9f/15.0f;

        for (unsigned int i = 0; i < frames.size(); ++i) {
            const frame& f = frames[i];
            if (f.number < d.start_frame)
                continue;
            else if (f.number > d.end_frame)
                break;

            if (f.ts >= 0 && f.ts < m_last_ts + min_frame_gap) continue;
            m_last_ts = f.ts;

            const auto& frame = frames.at(i);
            const auto imgFull = cv::imread(frame.color_path);
            cv::Mat img;
            cv::pyrDown(imgFull, img);

            const auto tp_1 = std::chrono::steady_clock::now();

            if (!img.empty() && (i % frame_skip == 0)) {
                // input the current frame and estimate the camera pose
                SLAM.feed_monocular_frame(img, frame.ts, mask);
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            if (i % frame_skip == 0) {
                track_times.push_back(track_time);
            }

            // wait until the timestamp of the next frame
            if (!no_sleep && i < frames.size() - 1) {
                const auto wait_time = frames.at(i + 1).ts - (frame.ts + track_time);
                if (0.0 < wait_time) {
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
                }
            }

            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }

        // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
        if (auto_term) {
            viewer.request_terminate();
        }
#elif USE_SOCKET_PUBLISHER
        if (auto_term) {
            publisher.request_terminate();
        }
#endif
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs("track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto dataset_file_path = op.add<popl::Value<std::string>>("d", "dataset", "path to dataset folder containing images and log.csv");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
    auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
    auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "store a map database at this path after SLAM", "");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !dataset_file_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // load dataset
    dataset d(dataset_file_path->value());
    if (!d.load()) {
        spdlog::error("Unable to load dataset");
        return 1;
    }

    d.start_frame = std::max(d.start_frame, d.get_frames().front().number);
    d.end_frame = std::min(d.end_frame, d.get_frames().back().number);

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_tracking(cfg, vocab_file_path->value(), d, mask_img_path->value(),
                      frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
                      eval_log->is_set(), map_db_path->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
