/**
* Headless stereo-inertial driver for ORB-SLAM3, adapted from
* Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc of ORB-SLAM3
* (GPLv3, Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez
* Rodriguez, Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza).
* Changes: one sequence, viewer off, trajectory name argument, per-frame
* tracking-state counts on stdout. CLAHE as in the original.
*
* Usage: stereo_inertial_kb vocabulary settings cam0_dir cam1_dir times.txt imu.csv name
* Writes f_<name>.txt and kf_<name>.txt (EuRoC: t_ns tx ty tz qx qy qz qw, body in world)
* to the working directory.
*/
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <map>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>
#include "ImuTypes.h"

using namespace std;

static void LoadImages(const string &left, const string &right, const string &times,
                       vector<string> &vLeft, vector<string> &vRight, vector<double> &vT)
{
    ifstream f(times.c_str());
    string s;
    while (getline(f, s)) {
        if (s.empty() || s[0] == '#') continue;
        string item = s.substr(0, s.find(' '));
        vLeft.push_back(left + "/" + item + ".png");
        vRight.push_back(right + "/" + item + ".png");
        vT.push_back(stod(item) / 1e9);
    }
}

static void LoadIMU(const string &path, vector<double> &vT, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream f(path.c_str());
    string s;
    while (getline(f, s)) {
        if (s.empty() || s[0] == '#') continue;
        double d[7];
        size_t pos;
        int n = 0;
        while ((pos = s.find(',')) != string::npos && n < 6) {
            d[n++] = stod(s.substr(0, pos));
            s.erase(0, pos + 1);
        }
        d[6] = stod(s);
        vT.push_back(d[0] / 1e9);
        vGyro.push_back(cv::Point3f(d[1], d[2], d[3]));
        vAcc.push_back(cv::Point3f(d[4], d[5], d[6]));
    }
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        cerr << "Usage: stereo_inertial_kb vocabulary settings cam0_dir cam1_dir times.txt imu.csv name" << endl;
        return 1;
    }
    const string name = argv[7];
    vector<string> vLeft, vRight;
    vector<double> vTimesCam, vTimesImu;
    vector<cv::Point3f> vAcc, vGyro;
    LoadImages(argv[3], argv[4], argv[5], vLeft, vRight, vTimesCam);
    LoadIMU(argv[6], vTimesImu, vAcc, vGyro);
    const int nImages = vLeft.size(), nImu = vTimesImu.size();
    cout << "images: " << nImages << " imu: " << nImu << endl;
    if (nImages <= 0 || nImu <= 0) { cerr << "ERROR: no images or no IMU" << endl; return 1; }

    // First IMU sample to use: the last one at or before the first image (IMU starts first).
    int first_imu = 0;
    while (first_imu < nImu && vTimesImu[first_imu] <= vTimesCam[0]) first_imu++;
    first_imu = max(first_imu - 1, 0);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false, 0, name);
    const float imageScale = SLAM.GetImageScale();
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    vector<float> vTimesTrack(nImages, 0.f);
    map<int, int> stateCount;
    int lostEvents = 0, prevState = -2;

    for (int ni = 0; ni < nImages; ni++) {
        cv::Mat imLeft = cv::imread(vLeft[ni], cv::IMREAD_GRAYSCALE);
        cv::Mat imRight = cv::imread(vRight[ni], cv::IMREAD_GRAYSCALE);
        if (imLeft.empty() || imRight.empty()) {
            cerr << "ERROR: failed to load image " << vLeft[ni] << endl;
            return 1;
        }
        if (imageScale != 1.f) {
            cv::Size sz(imLeft.cols * imageScale, imLeft.rows * imageScale);
            cv::resize(imLeft, imLeft, sz);
            cv::resize(imRight, imRight, sz);
        }
        clahe->apply(imLeft, imLeft);
        clahe->apply(imRight, imRight);
        const double tframe = vTimesCam[ni];

        vImuMeas.clear();
        if (ni > 0) {
            while (first_imu < nImu && vTimesImu[first_imu] <= tframe) {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                                                         vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                                                         vTimesImu[first_imu]));
                first_imu++;
            }
        }

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);
        auto t2 = chrono::steady_clock::now();
        const double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        const int state = SLAM.GetTrackingState();
        stateCount[state]++;
        if (state == 4 && prevState != 4) lostEvents++;  // Tracking::LOST
        prevState = state;

        // Real-time pacing as in the original example, so local mapping keeps up.
        double T = 0;
        if (ni < nImages - 1) T = vTimesCam[ni + 1] - tframe;
        else if (ni > 0) T = tframe - vTimesCam[ni - 1];
        if (ttrack < T) usleep((T - ttrack) * 1e6);
    }

    SLAM.Shutdown();
    SLAM.SaveTrajectoryEuRoC("f_" + name + ".txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("kf_" + name + ".txt");

    sort(vTimesTrack.begin(), vTimesTrack.end());
    double total = 0;
    for (float t : vTimesTrack) total += t;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << total / nImages << endl;
    cout << "TRACKING_STATES";
    for (auto &kv : stateCount) cout << " " << kv.first << ":" << kv.second;
    cout << endl << "LOST_EVENTS " << lostEvents << endl;
    return 0;
}
