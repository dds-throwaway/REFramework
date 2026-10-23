#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/callback_sink.h>

// Keeps the default logger's mutex out of a frozen world.
//
// A `utility::ThreadSuspender` (kananlib) takes the PEB + loader locks and then suspends every other
// thread, logging one line per thread *from inside that frozen window*:
//
//     SPDLOG_INFO("Suspending {}", (uint32_t)te.th32ThreadID);      // kananlib src/Thread.cpp
//     state->suspended = SuspendThread(thread_handle) != (DWORD)-1;
//
// SPDLOG_INFO goes to `spdlog::default_logger_raw()` - the same logger, and the same sink mutex, that
// REFramework and its hook callbacks use. A thread suspended while inside a log call therefore holds that
// mutex, and the suspender's *next* log call waits on a thread it has already frozen: the PEB and loader
// locks stay held, every suspended thread stays suspended, and the game wedges - no crash, no menu, and
// nothing outside can release the loader lock (it belongs to a critical section owned by a stuck thread).
// The dependency is not ours to change, so instead nothing may touch that mutex while frozen.
//
// For the lifetime of this object the default logger is a relay: log calls during the freeze go to a sink
// with its own uncontended mutex, and are replayed to the real logger once the world is running again. The
// per-thread suspension lines are the dependency's noise and would otherwise be thousands of lines per
// run, so they are counted and reported as one line.
//
// Declare it *before* the suspender in the same scope, so it is constructed first and destroyed last
// (destruction is reversed, and the real logger must only come back once the world is running):
//
//     WorldFreezeLog freeze_log;
//     utility::ThreadSuspender suspender{};
//     ...
//     suspender.resume();
//
// Not covered by this: any *other* lock a frozen thread may hold. Work done inside the frozen window
// still allocates and takes CRT/heap locks, which is the remaining reason to keep that window short.
class WorldFreezeLog {
public:
    WorldFreezeLog() {
        auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
            [this](const spdlog::details::log_msg& msg) { capture(msg); });

        m_relay = std::make_shared<spdlog::logger>("world_freeze_relay", std::move(sink));
        m_relay->set_level(spdlog::level::trace);
        m_relay->flush_on(spdlog::level::off);

        m_previous = spdlog::default_logger();

        if (m_previous != nullptr) {
            spdlog::set_default_logger(m_relay);
        }
    }

    ~WorldFreezeLog() {
        if (m_previous == nullptr) {
            return;
        }

        spdlog::set_default_logger(m_previous); // the world is running again by the time this is destroyed

        try {
            replay();
        } catch (...) {
            // A logger that throws while leaving scope must not take the process with it.
        }
    }

    WorldFreezeLog(const WorldFreezeLog&) = delete;
    WorldFreezeLog& operator=(const WorldFreezeLog&) = delete;

private:
    struct Entry {
        spdlog::source_loc source{};
        spdlog::level::level_enum level{ spdlog::level::info };
        std::string payload{};
    };

    static bool is_suspension_noise(const spdlog::details::log_msg& msg) {
        const auto* filename = msg.source.filename;

        if (filename == nullptr) {
            return false;
        }

        // The dependency's per-thread lines, matched on both file and text so unrelated Thread.cpp logs
        // are not swallowed.
        const std::string_view file{ filename };
        const std::string_view text{ msg.payload.data(), msg.payload.size() };

        return (file == "Thread.cpp" || file.ends_with("Thread.cpp")) &&
            (text.starts_with("Suspending ") || text.starts_with("Resuming "));
    }

    void capture(const spdlog::details::log_msg& msg) {
        std::scoped_lock _{ m_mutex };

        if (is_suspension_noise(msg)) {
            ++m_suspended_lines;
            return;
        }

        m_entries.push_back({ msg.source, msg.level, std::string{ msg.payload.data(), msg.payload.size() } });
    }

    void replay() {
        std::vector<Entry> entries{};
        size_t suspended_lines = 0;

        {
            std::scoped_lock _{ m_mutex };
            entries.swap(m_entries);
            suspended_lines = m_suspended_lines;
            m_suspended_lines = 0;
        }

        if (suspended_lines != 0) {
            m_previous->log(spdlog::level::info, "world freeze: {} thread suspension/resume lines suppressed",
                suspended_lines);
        }

        for (const auto& entry : entries) {
            m_previous->log(entry.source, entry.level, "{}", entry.payload);
        }
    }

    std::shared_ptr<spdlog::logger> m_previous{};
    std::shared_ptr<spdlog::logger> m_relay{};
    std::mutex m_mutex{};
    std::vector<Entry> m_entries{};
    size_t m_suspended_lines{ 0 };
};
