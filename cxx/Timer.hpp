#pragma once

#include <chrono>
#include <string>
#include <iostream>

class Timer {
public:
    Timer() {}

    void begin() {
        beginTime = std::chrono::steady_clock::now();
    }

    uint64_t end() {
        using namespace std::chrono;
        auto endTime = steady_clock::now();
        return duration_cast<nanoseconds>(endTime - beginTime).count();
    }

private:
    std::chrono::steady_clock::time_point beginTime;
};

inline void PrintTime(uint64_t nano) {
    static const std::string units[4] = { "ns", "us", "ms", "s" };
    auto u = 0u;
    auto fTime = float(nano);
    while (u < 3 && fTime > 1000) {
        u++;
        fTime /= 1000;
    }
    printf("Time: %f %s\n", fTime, units[u].c_str());
}

#define TIME_CODE(code, round) { \
    auto _timeSum = 0ull; \
    for (auto _ = 0u; _ < round; _++) { \
        Timer timer; \
        timer.begin(); \
        code \
        _timeSum += timer.end(); \
    } \
    PrintTime(_timeSum / round); \
}
