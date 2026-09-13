#include <unordered_map>
#include <cwchar>
#include <algorithm>
#include <unordered_set>
#include <shared_mutex>
#include <iomanip>
#include <regex>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <immintrin.h>
#include <sstream>
#include <string_view>

#include <asmjit/asmjit.h>
#include <asmjit/x86/x86assembler.h>

#include "utility/Exceptions.hpp"
#include "utility/Module.hpp"
#include "utility/Scan.hpp"
#include "utility/Emulation.hpp"
#include <bdshemu.h>

// Windows.h arrives via the utility headers above; TlHelp32 requires it to come first.
#include <TlHelp32.h>

#include "sdk/RETypeDB.hpp"
#include <sdk/GameIdentity.hpp>

#include "Hooks.hpp"

#include "IntegrityCheckBypass.hpp"
#include "DisasmUtils.hpp"

struct IntegrityCheckPattern {
    std::string pat{};
    uint32_t offset{};
};

namespace {

// The game's integrity protection trips via DebugBreak (STATUS_BREAKPOINT). Hooking it lets us log
// the exact call site and callstack, which is the "injection path" the protection uses. This is
// diagnostic: the original DebugBreak is still called so behaviour is unchanged.
std::unique_ptr<FunctionHook> s_debug_break_hook{};
std::atomic<bool> s_dumped_debug_break_callstack{false};

void WINAPI debug_break_hook() {
    spdlog::warn("[IntegrityCheckBypass]: DebugBreak called from 0x{:X}", (uintptr_t)_ReturnAddress());

    if (!s_dumped_debug_break_callstack.exchange(true)) {
        utility::exceptions::dump_callstack(nullptr);
    }

    if (s_debug_break_hook != nullptr) {
        s_debug_break_hook->get_original<decltype(debug_break_hook)>()();
    }
}

using RtlAddVectoredExceptionHandler_t = PVOID(NTAPI*)(ULONG, PVECTORED_EXCEPTION_HANDLER);

// First-chance logger. The unhandled exception filter never fires for this crash, so catch the
// exception (access violation / inline int3 / illegal instruction) before anything can swallow it
// and log the faulting RIP + callstack. Registered via ntdll directly so it bypasses REFramework's
// own AddVectoredExceptionHandler filter.
// Resolve a raw code address to the function entrypoint containing it. Uses the game's exception
// directory (.pdata, RVA 0x202BA000) via kananlib. Addresses inside the obfuscated .udata region
// have no unwind entries and will not resolve.
static std::optional<uintptr_t> kananlib_function_entrypoint(uintptr_t addr) {
    if (const auto entry = utility::find_function_start_unwind(addr)) {
        return entry;
    }

    return utility::find_function_start(addr);
}

// True if the live bytes at `addr` differ from the on-disk module image, i.e. something patched it
// at runtime. Returns the original (disk) bytes of the patched prefix.
static std::vector<uint8_t> kananlib_patched_bytes(uintptr_t addr) {
    if (const auto original = utility::get_original_bytes(addr)) {
        return *original;
    }

    return {};
}

static std::string describe_address(uintptr_t addr) {
    const auto module = utility::get_module_within(addr);
    if (!module) {
        return fmt::format("0x{:X} (no module)", addr);
    }

    std::string out = fmt::format("0x{:X} (module+0x{:X}", addr, addr - (uintptr_t)*module);

    if (const auto entry = kananlib_function_entrypoint(addr)) {
        out += fmt::format(", fn+0x{:X}", addr - *entry);
    } else {
        out += ", no fn entry";
    }

    return out + ")";
}

// Log every qword in [rsp, rsp + depth*8) that points into a module. Used on the faulting stack
// (return addresses) and, at a hook site, on the game's own stack: a mid-hook's trampoline is where
// RtlCaptureStackBackTrace stops, so the game frames below it are only visible by walking
// context.rsp by hand.
static void log_stack_scan(uintptr_t rsp, size_t depth, std::string_view tag) {
    if (rsp == 0 || IsBadReadPtr((void*)rsp, depth * sizeof(uintptr_t))) {
        return;
    }

    const auto stack = (const uintptr_t*)rsp;

    for (size_t i = 0; i < depth; ++i) {
        const auto value = stack[i];

        if (value < 0x10000 || value > 0x7FFFFFFFFFFF) {
            continue;
        }

        if (utility::get_module_within(value)) {
            spdlog::error("[IntegrityCheckBypass]:     {} stack[{}] = {}", tag, i, describe_address(value));
        }
    }
}

// The fatal fault is a `ret` to a zeroed return slot, and RSP at the fault is byte-identical on
// every run (0x1850C3E0), so the destroyed slot has a FIXED address. That makes the store that
// destroys it catchable with a hardware data-write breakpoint instead of guessing at gadgets:
// `mov qword [rsp],0` alone occurs 7348 times in this image (7344 of them in .udata), all part of
// the obfuscator's stack-spoofing idiom, so pattern-matching the destroyer is a dead end.
//
// DR0..DR3 cover 8 bytes each, which spans the whole window visible in the fault frame:
//   [rsp-32] = the ud2 decoy, [rsp-24]/[rsp-16] locals, [rsp-8] the wiped return slot.
// A hit is a trap AFTER the store, so RIP is the instruction following it.
// The frame sits at a fixed offset from the thread's stack base: RSP at the fault is
// StackBase - 0x3C20 on every run (0x1850C3E0 with StackBase 0x18510000, 0x19BDC3E0 with StackBase
// 0x19BE0000). An absolute address is therefore wrong the moment anything shifts the layout -- which
// is exactly what happened once the early phase started creating threads, and it silently cost a run.
// Derive the window per thread instead.
static constexpr uintptr_t STACK_WATCH_FROM_BASE = 0x3C40; // ..-0x20, i.e. [R0-24, R0]
static constexpr size_t STACK_WATCH_SIZE = 0x20;
static constexpr uintptr_t STACK_WATCH_DR7 = 0x99990055; // 4x local-enable, RW=write, LEN=8 bytes
static constexpr uint32_t STACK_WATCH_MAX_HITS = 128;

// Minimal THREAD_BASIC_INFORMATION (ThreadBasicInformation == 0); winternl.h is not pulled in here.
struct ThreadBasicInfo {
    NTSTATUS exit_status;
    void* teb;
    uintptr_t unique_process;
    uintptr_t unique_thread;
    uintptr_t affinity_mask;
    int32_t priority;
    int32_t base_priority;
};

// NT_TIB.StackBase lives at +8 in the TEB.
static uintptr_t stack_base_of(void* teb) {
    return teb != nullptr ? *(const uintptr_t*)((uintptr_t)teb + 8) : 0;
}

static uintptr_t stack_watch_base_of(void* teb) {
    const auto base = stack_base_of(teb);

    return base > STACK_WATCH_FROM_BASE ? base - STACK_WATCH_FROM_BASE : 0;
}

static uintptr_t stack_watch_base_here() {
    return stack_watch_base_of(NtCurrentTeb());
}

static std::atomic<uint32_t> s_stack_watch_hits{0};
static std::atomic<bool> s_stack_watch_disarmed{false};
static std::vector<std::unique_ptr<Patch>> s_stack_watch_patches{};

// Wine backs hardware breakpoints with ptrace, which only exists while the thread is being traced by
// something. A silent "0 hits" is therefore ambiguous -- no write happened, or the debug registers
// were never programmed -- so prove the mechanism on the calling thread before trusting a negative.
static volatile uint64_t s_dr_selftest_slot[2]{};
static std::atomic<bool> s_dr_selftest_active{false};
static std::atomic<bool> s_dr_selftest_hit{false};
static constexpr uintptr_t DR_SELFTEST_DR7 = 0x00090001; // L0, R/W0=write, LEN0=8 bytes

// Nothing logged before REFramework's constructor runs reaches the log file -- the file sink is
// installed there (REFramework.cpp:259/270), which is why `[Thread] startup_thread TID` never shows
// up. The early-phase work below has to run before that point, so buffer its output and flush it
// once the sink exists.
static std::atomic<bool> s_early_phase{true};
static std::vector<std::string> s_deferred_early_log{};

// A raw, flushed, always-on early log. The main log file is only created by the REFramework
// constructor, so anything that kills the process during the early phase leaves no log at all -- which
// has now happened twice. This writes with Win32 directly (no CRT, no heap) and flushes per line.
static HANDLE s_early_log_file = INVALID_HANDLE_VALUE;

static void early_log_raw(const std::string& line) {
    if (s_early_log_file == INVALID_HANDLE_VALUE) {
        return;
    }

    const auto text = line + "\r\n";
    DWORD written = 0;

    WriteFile(s_early_log_file, text.data(), (DWORD)text.size(), &written, nullptr);
    FlushFileBuffers(s_early_log_file);
}

static void early_log_open() {
    if (s_early_log_file != INVALID_HANDLE_VALUE) {
        return;
    }

    wchar_t module_path[MAX_PATH]{};

    if (GetModuleFileNameW(nullptr, module_path, MAX_PATH) == 0) {
        return;
    }

    std::wstring path{module_path};
    const auto slash = path.find_last_of(L"\\/");

    if (slash == std::wstring::npos) {
        return;
    }

    path = path.substr(0, slash + 1) + L"reframework_early_log.txt";

    s_early_log_file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

static void diag_log(std::string message) {
    early_log_raw(message);

    if (s_early_phase.load(std::memory_order_relaxed)) {
        if (s_deferred_early_log.size() < 1024) {
            s_deferred_early_log.push_back(std::move(message));
        }

        return;
    }

    spdlog::error("[IntegrityCheckBypass]: {}", message);
}

static void flush_early_log() {
    s_early_phase = false;

    for (const auto& message : s_deferred_early_log) {
        spdlog::error("[IntegrityCheckBypass]: (early) {}", message);
    }

    spdlog::info("[IntegrityCheckBypass]: flushed {} buffered early diagnostic line(s).", s_deferred_early_log.size());
    s_deferred_early_log.clear();
}

// A data-breakpoint trap is EXCEPTION_SINGLE_STEP. Without the vectored handler installed it is an
// unhandled exception and kills the process on the spot -- which is exactly what the DR self-test
// did when it ran before the handler existed. Nothing may raise a trap before this is true.
static std::atomic<bool> s_first_chance_logger_installed{false};

// Debug registers are per-thread, so a thread created after arming is not covered. Track what has
// been armed and re-arm only the newcomers, so a background loop is cheap.
static std::unordered_set<DWORD> s_armed_tids{};
static std::mutex s_armed_tids_mutex{};

static void run_dr_selftest() {
    if (!s_first_chance_logger_installed.load(std::memory_order_relaxed)) {
        diag_log("DR self-test skipped: the vectored handler is not installed, so a single-step trap would be fatal");
        return;
    }

    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;

    if (!GetThreadContext(GetCurrentThread(), &ctx)) {
        diag_log("DR self-test: GetThreadContext failed");
        return;
    }

    ctx.Dr0 = (uintptr_t)&s_dr_selftest_slot[0];
    ctx.Dr1 = ctx.Dr2 = ctx.Dr3 = 0;
    ctx.Dr7 = DR_SELFTEST_DR7;

    if (!SetThreadContext(GetCurrentThread(), &ctx)) {
        diag_log("DR self-test: SetThreadContext failed");
        return;
    }

    s_dr_selftest_active = true;
    s_dr_selftest_hit = false;
    s_dr_selftest_slot[0] = 0x1122334455667788ULL;
    s_dr_selftest_active = false;

    const auto hit = s_dr_selftest_hit.load();

    ctx.Dr0 = 0;
    ctx.Dr7 = 0;
    SetThreadContext(GetCurrentThread(), &ctx);

    if (hit) {
        diag_log("DR self-test OK -- hardware write watchpoints are live");
    } else {
        diag_log("DR self-test FAILED -- the OS/Wine did not program the debug registers, so a negative"
                 " stack-watch result proves nothing. Use an external debugger: winedbg --gdb +"
                 " `watch *(long long*)0x1850C3D8`");
    }
}

// Arm a write watchpoint on every other thread in the process. Called while REFramework's
// ThreadSuspender already has them frozen, so the +1/-1 suspend count here is a no-op on them.
static void arm_stack_write_watchpoints(bool report = true) {
    using NtQueryInformationThread_t = NTSTATUS(NTAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
    static const auto nt_query_information_thread =
        (NtQueryInformationThread_t)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread");

    if (nt_query_information_thread == nullptr) {
        diag_log("could not resolve NtQueryInformationThread");
        return;
    }

    const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);

    if (snapshot == INVALID_HANDLE_VALUE) {
        diag_log("could not snapshot threads to arm stack watchpoints");
        return;
    }

    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);

    const auto pid = GetCurrentProcessId();
    const auto self = GetCurrentThreadId();
    uint32_t armed = 0;

    for (auto ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) {
            continue;
        }

        {
            std::scoped_lock _{s_armed_tids_mutex};

            if (!s_armed_tids.emplace(entry.th32ThreadID).second) {
                continue;
            }
        }

        const auto thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE, entry.th32ThreadID);

        if (thread == nullptr) {
            continue;
        }

        if (SuspendThread(thread) == (DWORD)-1) {
            CloseHandle(thread);
            continue;
        }

        // Per-thread window, derived from this thread's own stack base.
        ThreadBasicInfo info{};

        if (nt_query_information_thread(thread, 0 /*ThreadBasicInformation*/, &info, sizeof(info), nullptr) < 0) {
            ResumeThread(thread);
            CloseHandle(thread);
            continue;
        }

        const auto watch = stack_watch_base_of(info.teb);
        const auto stack_base = stack_base_of(info.teb);

        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;

        uintptr_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        auto sampled = false;

        if (watch != 0 && GetThreadContext(thread, &ctx)) {
            ctx.Dr0 = watch;
            ctx.Dr1 = watch + 8;
            ctx.Dr2 = watch + 16;
            ctx.Dr3 = watch + 24;
            ctx.Dr7 = STACK_WATCH_DR7;

            if (SetThreadContext(thread, &ctx)) {
                ++armed;

                const auto window = (const uintptr_t*)watch;

                if (!IsBadReadPtr((void*)watch, STACK_WATCH_SIZE)) {
                    w0 = window[0];
                    w1 = window[1];
                    w2 = window[2];
                    w3 = window[3];
                    sampled = true;
                }
            }
        }

        ResumeThread(thread);
        CloseHandle(thread);

        // Log OUTSIDE the suspend/resume region: allocating (fmt, std::string, the deferred buffer)
        // while another thread is frozen can deadlock on a heap lock it is holding.
        if (sampled) {
            diag_log(fmt::format("  TID {} StackBase 0x{:X} window {:016X} {:016X} {:016X} {:016X}", entry.th32ThreadID, stack_base,
                w0, w1, w2, w3));
        }
    }

    CloseHandle(snapshot);

    if (!report) {
        return;
    }

    diag_log(fmt::format("armed stack write watchpoints on {} new thread(s); window = StackBase-0x{:X}..-0x{:X}", armed,
        STACK_WATCH_FROM_BASE, STACK_WATCH_FROM_BASE - STACK_WATCH_SIZE));

    // This thread's own window, so there is always a sample of what the frame looks like at arm time
    // even if the crashing thread did not exist yet.
    const auto own_watch = stack_watch_base_here();

    if (own_watch != 0 && !IsBadReadPtr((void*)own_watch, STACK_WATCH_SIZE)) {
        const auto window = (const uintptr_t*)own_watch;

        diag_log(fmt::format("own window at arm time (StackBase 0x{:X}): {:016X} {:016X} {:016X} {:016X}", stack_base_of(NtCurrentTeb()),
            window[0], window[1], window[2], window[3]));
    }

    // This thread is the one doing the arming, so it is not in the loop above; test on it.
    run_dr_selftest();
}

static void log_stack_watch_hit(EXCEPTION_POINTERS* ei) {
    const auto& ctx = *ei->ContextRecord;
    const auto rip = (uintptr_t)ctx.Rip;
    const auto hit = s_stack_watch_hits.fetch_add(1, std::memory_order_relaxed);
    const auto watch = stack_watch_base_here();
    const auto watched = (const uintptr_t*)watch;

    if (hit < STACK_WATCH_MAX_HITS) {
        std::string behind{};

        if (rip >= 16 && !IsBadReadPtr((void*)(rip - 16), 16)) {
            const auto bytes = (const uint8_t*)(rip - 16);

            for (size_t i = 0; i < 16; ++i) {
                behind += fmt::format("{:02X} ", bytes[i]);
            }
        }

        uintptr_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;

        if (watch != 0 && !IsBadReadPtr((void*)watch, STACK_WATCH_SIZE)) {
            w0 = watched[0];
            w1 = watched[1];
            w2 = watched[2];
            w3 = watched[3];
        }

        diag_log(fmt::format("stack watch hit #{}: writer RIP 0x{:X} ({}) bytes before: {} | window: {:016X} {:016X} {:016X} {:016X}",
            hit, rip, describe_address(rip), behind, w0, w1, w2, w3));
        // The destroyer's primitive. NOP it in place: the gadget keeps its control flow, the return
        // slot survives, and if this is the real culprit the crash goes away on the next ret.
        static constexpr uint8_t mov_rsp_zero[] = {0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00};

        if (rip >= sizeof(mov_rsp_zero) && !IsBadReadPtr((void*)(rip - sizeof(mov_rsp_zero)), sizeof(mov_rsp_zero)) &&
            memcmp((const void*)(rip - sizeof(mov_rsp_zero)), mov_rsp_zero, sizeof(mov_rsp_zero)) == 0) {
            s_stack_watch_patches.emplace_back(
                Patch::create(rip - sizeof(mov_rsp_zero), {0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90}, true));

            diag_log(fmt::format("NOP'd `mov qword [rsp],0` at 0x{:X} -- return slot preserved", rip - sizeof(mov_rsp_zero)));
        }
    }

    // Every write in the window traps. Bail out before that turns into a stall if it is a hot path.
    if (hit + 1 >= STACK_WATCH_MAX_HITS && !s_stack_watch_disarmed.exchange(true)) {
        ei->ContextRecord->Dr0 = 0;
        ei->ContextRecord->Dr1 = 0;
        ei->ContextRecord->Dr2 = 0;
        ei->ContextRecord->Dr3 = 0;
        ei->ContextRecord->Dr7 = 0;

        diag_log(fmt::format("stack watchpoints disarmed after {} hits", STACK_WATCH_MAX_HITS));
    }
}

LONG CALLBACK first_chance_exception_logger(EXCEPTION_POINTERS* ei) {
    if (ei == nullptr || ei->ExceptionRecord == nullptr || ei->ContextRecord == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    const auto code = ei->ExceptionRecord->ExceptionCode;

    // Data breakpoint from arm_stack_write_watchpoints(). Traps after the store, so RIP is already
    // past the writer.
    if (code == EXCEPTION_SINGLE_STEP) {
        // Self-test: our own store to a known slot.
        if (s_dr_selftest_active.load() && ei->ContextRecord->Dr7 == DR_SELFTEST_DR7) {
            s_dr_selftest_hit = true;
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        // Only ours: Wine/the game can raise single-step for other reasons (TF, internal stepping).
        if (ei->ContextRecord->Dr7 == STACK_WATCH_DR7) {
            log_stack_watch_hit(ei);
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        return EXCEPTION_CONTINUE_SEARCH;
    }

    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_BREAKPOINT && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_IN_PAGE_ERROR && code != 0xC0000409 /*STATUS_STACK_BUFFER_OVERRUN*/) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    const auto rip = (uintptr_t)ei->ContextRecord->Rip;

    // Dedupe by faulting RIP instead of a hard cap: REFramework's own speculative memory probing
    // fires the same few AVs over and over and previously burned the 32-entry budget before the
    // real crash. Each unique RIP is logged once.
    constexpr size_t max_unique_rips = 128;
    static std::atomic<uintptr_t> s_seen_rips[max_unique_rips];
    static std::atomic<size_t> s_seen_rip_count{0};

    const auto seen_count = s_seen_rip_count.load(std::memory_order_relaxed);

    // Never drop the fatal case. The handler is now installed during the early phase and sees a lot
    // more traffic before the crash, so the dedup budget could otherwise be spent by then.
    if (rip != 0) {
        for (size_t i = 0; i < seen_count && i < max_unique_rips; ++i) {
            if (s_seen_rips[i].load(std::memory_order_relaxed) == rip) {
                return EXCEPTION_CONTINUE_SEARCH;
            }
        }

        if (seen_count >= max_unique_rips) {
            return EXCEPTION_CONTINUE_SEARCH;
        }

        s_seen_rips[seen_count].store(rip, std::memory_order_relaxed);
        s_seen_rip_count.store(seen_count + 1, std::memory_order_relaxed);
    }

    spdlog::error("[IntegrityCheckBypass]: First-chance exception 0x{:X} at 0x{:X}, RIP 0x{:X}, RSP 0x{:X}", code,
        (uintptr_t)ei->ExceptionRecord->ExceptionAddress, rip, (uintptr_t)ei->ContextRecord->Rsp);

    if (code == EXCEPTION_ACCESS_VIOLATION && ei->ExceptionRecord->NumberParameters >= 2) {
        const auto av_type = ei->ExceptionRecord->ExceptionInformation[0]; // 0=read 1=write 8=execute
        const auto av_addr = (uintptr_t)ei->ExceptionRecord->ExceptionInformation[1];

        spdlog::error("[IntegrityCheckBypass]:     AV {} target 0x{:X}",
            av_type == 0 ? "read" : (av_type == 1 ? "write" : (av_type == 8 ? "execute" : "?")), av_addr);
    }

    // For a jump/call through a NULL pointer the registers are the interesting part: the target
    // (RAX/reg) is 0 and RCX/RDX/R8/R9 hold the call arguments (usually `this` + args).
    if (rip == 0) {
        const auto& ctx = *ei->ContextRecord;

        spdlog::error("[IntegrityCheckBypass]:     regs: RAX 0x{:X} RBX 0x{:X} RCX 0x{:X} RDX 0x{:X} RSI 0x{:X} RDI 0x{:X} R8 0x{:X} R9 "
                      "0x{:X} RBP 0x{:X}",
            ctx.Rax, ctx.Rbx, ctx.Rcx, ctx.Rdx, ctx.Rsi, ctx.Rdi, ctx.R8, ctx.R9, ctx.Rbp);

        // A `ret` to 0 leaves the destroyed slot just below RSP; a `call` through 0 leaves the
        // caller's return address at [RSP]. This tells the two apart.
        const auto fault_rsp = (uintptr_t)ctx.Rsp;

        if (fault_rsp >= 0x20 && !IsBadReadPtr((void*)(fault_rsp - 0x20), 0x28)) {
            const auto below = (const uintptr_t*)(fault_rsp - 0x20);

            spdlog::error("[IntegrityCheckBypass]:     below RSP: [rsp-8] 0x{:X} [rsp-16] 0x{:X} [rsp-24] 0x{:X} [rsp-32] 0x{:X}",
                below[3], below[2], below[1], below[0]);
        }

        // The watched window at fault time, to compare against `window at arm time`.
        const auto watch = stack_watch_base_here();

        if (watch != 0 && !IsBadReadPtr((void*)watch, STACK_WATCH_SIZE)) {
            const auto window = (const uintptr_t*)watch;

            spdlog::error("[IntegrityCheckBypass]:     window at fault: {:016X} {:016X} {:016X} {:016X}", window[0], window[1],
                window[2], window[3]);
        }

        // Everything the VM left in registers plus a raw stack window. The stack is destroyed and
        // the registers are the only other state, so capture both verbatim -- there is no second
        // chance to look at this process.
        spdlog::error("[IntegrityCheckBypass]:     regs2: R10 0x{:X} R11 0x{:X} R12 0x{:X} R13 0x{:X} R14 0x{:X} R15 0x{:X} RSP 0x{:X} RIP 0x{:X} EFLAGS 0x{:X}",
            ctx.R10, ctx.R11, ctx.R12, ctx.R13, ctx.R14, ctx.R15, ctx.Rsp, ctx.Rip, ctx.EFlags);

        if (fault_rsp != 0 && !IsBadReadPtr((void*)fault_rsp, 0x60 * sizeof(uintptr_t))) {
            const auto stack = (const uintptr_t*)fault_rsp;

            for (size_t row = 0; row < 0x60; row += 4) {
                spdlog::error("[IntegrityCheckBypass]:     raw stack[{:3}..{:3}] {:016X} {:016X} {:016X} {:016X}", row, row + 3,
                    stack[row], stack[row + 1], stack[row + 2], stack[row + 3]);
            }
        }
    }

    // Identify the faulting thread. ThreadSuspender freezes *every* thread in the process (including
    // REFramework's own workers), so knowing whether the crash is on a dinput8-owned thread or a
    // game-created one changes the diagnosis entirely. The thread's start address resolves that:
    // start inside the REFramework module => REFramework thread, otherwise the game owns it.
    {
        const auto tid = GetCurrentThreadId();
        uintptr_t thread_start = 0;

        // ThreadQuerySetWin32StartAddress == 9. Resolved dynamically to avoid winternl.h dependencies.
        using NtQueryInformationThread_t = NTSTATUS(NTAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
        static const auto nt_query_information_thread =
            (NtQueryInformationThread_t)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread");

        if (const auto thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid)) {
            if (nt_query_information_thread != nullptr) {
                nt_query_information_thread(thread, 9, &thread_start, sizeof(thread_start), nullptr);
            }
            CloseHandle(thread);
        }

        const auto self = utility::get_module_within((uintptr_t)&first_chance_exception_logger);

        if (thread_start == 0) {
            spdlog::error("[IntegrityCheckBypass]:     thread TID {} has no Win32 start address (CRT/game-created); ownership unknown", tid);
        } else if (const auto module = utility::get_module_within(thread_start)) {
            const auto owner = (self && *module == *self) ? " <-- REFramework thread" : " <-- game thread";

            spdlog::error("[IntegrityCheckBypass]:     thread TID {} started at {}{}", tid, describe_address(thread_start), owner);

            if (const auto patched = kananlib_patched_bytes(thread_start); !patched.empty()) {
                std::string bytes{};
                for (const auto b : patched) {
                    bytes += fmt::format("{:02X} ", b);
                }

                spdlog::error("[IntegrityCheckBypass]:     thread entry patched at runtime: {} ({} bytes)", bytes, patched.size());
            }
        } else {
            spdlog::error("[IntegrityCheckBypass]:     thread TID {} started at 0x{:X}{}", tid, thread_start,
                (thread_start == 0) ? " (unknown / started via std::thread)" : "");
        }
    }

    // The faulting call's return address lives on the stack when RIP itself is garbage/0 (a call
    // through a NULL function pointer). dump_callstack() cannot see it because it is not part of the
    // faulting frame chain.
    const auto rsp = (uintptr_t)ei->ContextRecord->Rsp;

    if (rsp != 0 && !IsBadReadPtr((void*)rsp, 0x100 * sizeof(uintptr_t))) {
        const auto stack = (const uintptr_t*)rsp;

        spdlog::error("[IntegrityCheckBypass]:     stack: 0x{:X} 0x{:X} 0x{:X} 0x{:X} 0x{:X} 0x{:X} 0x{:X} 0x{:X}", stack[0], stack[1],
            stack[2], stack[3], stack[4], stack[5], stack[6], stack[7]);

        // Resolve every slot that lands inside a module; the return address of the faulty call is
        // deeper than the first 8 qwords when the frame has locals/a stack cookie in front of it.
        for (size_t i = 0; i < 0x100; ++i) {
            const auto value = stack[i];

            if (value < 0x10000 || value > 0x7FFFFFFFFFFF) {
                continue;
            }

            if (utility::get_module_within(value)) {
                spdlog::error("[IntegrityCheckBypass]:     stack[{}] = {}", i, describe_address(value));
            }
        }
    }

    // The hook trampoline's own frames are all dump_callstack() can see, so it is pure noise here.
    return EXCEPTION_CONTINUE_SEARCH;
}

void init_first_chance_exception_logger() {
    static bool s_initialized = false;

    if (s_initialized) {
        return;
    }

    s_initialized = true;

    const auto ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto rtl_add_veh =
        ntdll != nullptr ? (RtlAddVectoredExceptionHandler_t)GetProcAddress(ntdll, "RtlAddVectoredExceptionHandler") : nullptr;

    if (rtl_add_veh == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not resolve RtlAddVectoredExceptionHandler!");
        return;
    }

    if (rtl_add_veh(1, &first_chance_exception_logger) == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Failed to install first-chance exception logger!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Installed first-chance exception logger.");
    s_first_chance_logger_installed = true;
}

using NtTerminateProcess_t = NTSTATUS(NTAPI*)(HANDLE, NTSTATUS);

std::unique_ptr<FunctionHook> s_nt_terminate_process_hook{};

// The silent deaths do not go through kernelbase!TerminateProcess or RtlExitUserProcess. Hook the
// lowest-level ntdll wrapper so a direct ntdll call is still visible. Only dump a callstack when
// the game itself asked for termination.
NTSTATUS NTAPI nt_terminate_process_hook(HANDLE process, NTSTATUS exit_status) {
    const auto retaddr = (uintptr_t)_ReturnAddress();
    const auto caller = utility::get_module_within(retaddr);

    if (caller && *caller == utility::get_executable()) {
        spdlog::error(
            "[IntegrityCheckBypass]: NtTerminateProcess(0x{:X}, 0x{:X}) from 0x{:X}", (uintptr_t)process, (uint32_t)exit_status, retaddr);

        utility::exceptions::dump_callstack(nullptr);
    }

    if (s_nt_terminate_process_hook != nullptr) {
        return s_nt_terminate_process_hook->get_original<decltype(nt_terminate_process_hook)>()(process, exit_status);
    }

    return 0;
}

void init_terminate_process_watcher() {
    if (s_nt_terminate_process_hook != nullptr) {
        return;
    }

    const auto ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto nt_terminate = ntdll != nullptr ? GetProcAddress(ntdll, "NtTerminateProcess") : nullptr;

    if (nt_terminate == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find NtTerminateProcess to hook!");
        return;
    }

    s_nt_terminate_process_hook = std::make_unique<FunctionHook>((uintptr_t)nt_terminate, (uintptr_t)&nt_terminate_process_hook);

    if (!s_nt_terminate_process_hook->create()) {
        spdlog::error("[IntegrityCheckBypass]: Failed to hook NtTerminateProcess!");
        s_nt_terminate_process_hook.reset();
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked NtTerminateProcess for diagnostics.");
}

std::unique_ptr<FunctionHook> s_report_gsfailure_hook{};
std::unique_ptr<FunctionHook> s_self_report_gsfailure_hook{};

static void log_gsfailure(std::string_view who) {
    const auto ret_slot = (uintptr_t*)_AddressOfReturnAddress();

    spdlog::error("[IntegrityCheckBypass]: __report_gsfailure ({})! retaddr 0x{:X}", who, (uintptr_t)_ReturnAddress());
    spdlog::error("[IntegrityCheckBypass]: stack: 0x{:X} 0x{:X} 0x{:X} 0x{:X}", ret_slot[0], ret_slot[1], ret_slot[2], ret_slot[3]);

    utility::exceptions::dump_callstack(nullptr);
}

// The silent deaths are __fastfail (int 0x29): __security_check_cookie (0x14B1295F0) jmps to
// __report_gsfailure (0x14B15028C) on a stack cookie mismatch, which does int 0x29. That is the
// game's stack-destroyer response and is uncatchable by VEH/termination hooks. Hook both the game's
// and REFramework's own __report_gsfailure so the corrupted function and its callers are logged
// before the process dies.
void report_gsfailure_hook(uintptr_t cookie) {
    log_gsfailure("game");

    if (s_report_gsfailure_hook != nullptr) {
        s_report_gsfailure_hook->get_original<decltype(report_gsfailure_hook)>()(cookie);
    }
}

void self_report_gsfailure_hook(uintptr_t cookie) {
    log_gsfailure("reframework");

    if (s_self_report_gsfailure_hook != nullptr) {
        s_self_report_gsfailure_hook->get_original<decltype(self_report_gsfailure_hook)>()(cookie);
    }
}

static constexpr auto GSAFAILURE_SIG = "48 89 4C 24 08 48 83 EC 38 B9 17 00 00 00 FF 15";

void init_gsfailure_watcher(HMODULE game) {
    if (s_report_gsfailure_hook == nullptr) {
        if (const auto gsfailure = utility::scan(game, GSAFAILURE_SIG)) {
            s_report_gsfailure_hook = std::make_unique<FunctionHook>(*gsfailure, (uintptr_t)&report_gsfailure_hook);

            if (!s_report_gsfailure_hook->create()) {
                spdlog::error("[IntegrityCheckBypass]: Failed to hook game __report_gsfailure!");
                s_report_gsfailure_hook.reset();
            } else {
                spdlog::info("[IntegrityCheckBypass]: Hooked game __report_gsfailure.");
            }
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not find game __report_gsfailure!");
        }
    }

    if (s_self_report_gsfailure_hook == nullptr) {
        const auto self = utility::get_module_within((uintptr_t)&init_gsfailure_watcher);

        if (self && *self != game) {
            if (const auto gsfailure = utility::scan(*self, GSAFAILURE_SIG)) {
                s_self_report_gsfailure_hook = std::make_unique<FunctionHook>(*gsfailure, (uintptr_t)&self_report_gsfailure_hook);

                if (!s_self_report_gsfailure_hook->create()) {
                    spdlog::error("[IntegrityCheckBypass]: Failed to hook reframework __report_gsfailure!");
                    s_self_report_gsfailure_hook.reset();
                } else {
                    spdlog::info("[IntegrityCheckBypass]: Hooked reframework __report_gsfailure.");
                }
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find reframework __report_gsfailure!");
            }
        }
    }
}

void init_debug_break_watcher() {
    if (s_debug_break_hook != nullptr) {
        return;
    }

    const auto kernelbase = GetModuleHandleW(L"kernelbase.dll");
    const auto debug_break = kernelbase != nullptr ? GetProcAddress(kernelbase, "DebugBreak") : nullptr;

    if (debug_break == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find DebugBreak to hook!");
        return;
    }

    s_debug_break_hook = std::make_unique<FunctionHook>((uintptr_t)debug_break, (uintptr_t)&debug_break_hook);

    if (!s_debug_break_hook->create()) {
        spdlog::error("[IntegrityCheckBypass]: Failed to hook DebugBreak!");
        s_debug_break_hook.reset();
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked DebugBreak for anti-tamper callstack logging.");
}

} // anonymous namespace

std::shared_ptr<IntegrityCheckBypass> s_integrity_check_bypass_instance{nullptr};

std::shared_ptr<IntegrityCheckBypass>& IntegrityCheckBypass::get_shared_instance() {
    if (!s_integrity_check_bypass_instance) {
        s_integrity_check_bypass_instance = std::make_unique<IntegrityCheckBypass>();
    }
    return s_integrity_check_bypass_instance;
}

std::optional<std::string> IntegrityCheckBypass::on_initialize() {
    // Patterns for assigning or accessing of the integrity check boolean (RE3)
    // and for jumping past the integrity checks (RE8)
    // In RE8, the integrity checks cause a noticeable stutter as well.
    std::vector<IntegrityCheckPattern> possible_patterns{};

    const auto& gi = sdk::GameIdentity::get();
    if (gi.is_re3()) {
        possible_patterns = {
            /*
            cmp     qword ptr [rax+18h], 0
            cmovz   ecx, r15d
            mov     cs:bypass_integrity_checks, cl*/
            // Referenced above "steam_api64.dll"
            {"48 ? ? 18 00 41 ? ? ? 88 0D ? ? ? ?", 11}, 
            {"48 ? ? 18 00 0F ? ? 88 0D ? ? ? ? 49 ? ? ? 48", 10},
        };
    } else if (gi.is_re8()) {
        possible_patterns = {
            /*
            These are partially obfuscated and are within protected sections.
            The ja jumps past the checksum checks which cause very large stutters if they are ran.
            We'll replace the ja to always jump past the checksum checks.

            There are various patterns here because the code is obfuscated, there's an element of randomness per update.
            Lots of random junk code. Some instructions are obfuscated into multiple instructions as well.
            We're taking a shot in the dark here hoping that the obfuscated code
            stays generally the same past a game update.
            */

            /*
            sub     eax, ecx
            ja      NO_CHECKSUM_CHECKS1
            mov     eax, [rsp+whatever]
            */
            // app.PlayerCore.onDamage, app.EnemyCore.onDie2 (onDie2 gets called from onDie)
            {"29 c8 0f 87 ? ? ? ? 8b 84", 2},

            /*
            sub     eax, ecx
            ja      NO_CHECKSUM_CHECKS2
.           xor     eax, eax
            sub     eax, [rsp+whatever]
            */
            // app.PlayerCore.onDamage #2
            {"29 c8 0f 87 ? ? ? ? 31 C0 2B", 2},

            /*
            mov     eax, [rsp+whatever]
            sub     eax, ecx
            ja      NO_CHECKSUM_CHECKS3
            xor     eax, eax
            */
            // app.PlayerCore.onDamage #3, app.EnemyCore.onDie2 #2
            {"8b 84 ? ? ? ? ? 29 c8 0f 87 ? ? ? ?", 9},
            /* 
            There is another one inside of app.GlobalService.msgSceneTransition_afterDeactivate
            but didn't bother to patch it out. Reason being that it seems to only get called when loading is finished. 
            Maybe some more investigation is required here?
            */
            // The above function names can be found within il2cpp_dump.json, which is dumped with REFramework's "Dump SDK" button in developer mode.
        };
    }

    std::unordered_set<uintptr_t> already_patched{};

    const auto module_size = *utility::get_module_size(g_framework->get_module().as<HMODULE>());
    const auto module_end = g_framework->get_module() + module_size;

    for (auto& possible_pattern : possible_patterns) {
        spdlog::info("Scanning for {}", possible_pattern.pat);

        auto integrity_check_ref = utility::scan(g_framework->get_module().as<HMODULE>(), possible_pattern.pat);

        if (!integrity_check_ref) {
            continue;
        }

        if (gi.is_re3()) {
            m_bypass_integrity_checks = (bool*)utility::calculate_absolute(*integrity_check_ref + possible_pattern.offset);
        } else if (gi.is_re8()) {
            ignore_application_entries();

            while (integrity_check_ref) {
                const auto ja_instruction = *integrity_check_ref + possible_pattern.offset;

                if (already_patched.contains(ja_instruction)) {
                    spdlog::info("IntegrityCheckBypass: ja instruction at 0x{:X} already patched, continuing...", ja_instruction);
                    integrity_check_ref =
                        utility::scan(*integrity_check_ref + 1, module_end - (*integrity_check_ref + 1), possible_pattern.pat);
                    continue;
                }

                // Create a ja->jmp patch for bypassing the integrity check
                std::vector<uint8_t> patch_bytes{0xE9, 0x00, 0x00, 0x00, 0x00, 0x90};

                // Overwrite the target address with the original ja target. Add 1 byte because the new instruction is smaller.
                *(uint32_t*)&patch_bytes[1] = *(uint32_t*)(ja_instruction + 2) + 1;

                // Convert the uint8_t patch_bytes to int16_t vector
                std::vector<int16_t> patch_int16_bytes{};

                for (auto& patch_byte : patch_bytes) {
                    patch_int16_bytes.push_back(patch_byte);
                }

                // Log the patch address (ja_instruction) and bytes with spdlog
                spdlog::info("Patch address: 0x{:X}", ja_instruction);

                // Convert patch_bytes to hex string with stringstream and then log the string with spdlog
                std::stringstream ss;
                ss << std::hex << std::setfill('0');
                for (auto& patch_byte : patch_bytes) {
                    ss << std::setw(2) << (int)patch_byte << " ";
                }

                spdlog::info("Patch bytes: {}", ss.str());

                // Patch the bytes
                m_patches.emplace_back(Patch::create(ja_instruction, patch_int16_bytes));
                already_patched.emplace(ja_instruction);

                // Search for the next integrity check using the same pattern
                integrity_check_ref = utility::scan(*integrity_check_ref + 1, module_end - (*integrity_check_ref + 1), possible_pattern.pat);
            }

            // If we didn't find any integrity checks
            if (m_patches.empty()) {
                spdlog::info("Could not find any integrity checks to bypass!");
            }
        }
    }

    // These may be removed, so don't fail altogether
    /*if (m_bypass_integrity_checks == nullptr) {
        return "Failed to find IntegrityCheckBypass pattern";
    }*/

    if (gi.is_re3()) {
        spdlog::info("[{:s}]: bypass_integrity_checks: {:x}", get_name().data(), (uintptr_t)m_bypass_integrity_checks);
    }

    if (gi.is_mhrise()) {
    // this is pretty much what it was like finding this, you just gotta look a little closer!
    const auto very_cool_type = sdk::find_type_definition_by_fqn(0x83f09f47);
    static std::vector<Patch::Ptr> very_cool_patches{};

    auto find_method_by_hash = [](sdk::RETypeDefinition* t, size_t hash) -> sdk::REMethodDefinition* {
        for (auto& method : t->get_methods()) {
            if (utility::hash(method.get_name()) == hash) {
                return &method;
            }
        }

        return nullptr;
    };

    if (very_cool_type != nullptr) {
        auto patch_very_cool_method = [&](size_t hash) {
            const auto method = find_method_by_hash(very_cool_type, hash);

            if (method == nullptr) {
                spdlog::error("Could not find very cool method", hash);
                return false;
            }

            if (method->get_function() == nullptr) {
                spdlog::error("[{:s}]: Could not find very_cool_type::very_cool_method!", get_name().data());
                return false;
            }

            spdlog::info("[{:s}]: Patching very cool method!", get_name().data());
        
            very_cool_patches.emplace_back(Patch::create((uintptr_t)method->get_function(), { 0xB0, 0x00, 0xC3 }, true));

            return true;
        };

        patch_very_cool_method(0x21c27632fa7ba29b);
        patch_very_cool_method(0x49b943a462e8cf6a);
    } else {
        spdlog::error("[{:s}]: Could not find very_cool_type!", get_name().data());
    }

    const auto very_awesome_type = sdk::find_type_definition_by_fqn(0xce04a0c6);

    if (very_awesome_type != nullptr) {
        const auto very_awesome_method = find_method_by_hash(very_awesome_type, 0x9f79221341cfcb18);

        if (very_awesome_method != nullptr) {
            if (very_awesome_method->get_function() == nullptr) {
                spdlog::error("[{:s}]: Could not find very_awesome_type::very_awesome_method!", get_name().data());
                return Mod::on_initialize();
            }

            const auto very_awesome_call = utility::scan_opcode((uintptr_t)very_awesome_method->get_function(), 10, 0xE8);

            if (!very_awesome_call) {
                spdlog::error("[{:s}]: Could not find very_awesome_call!", get_name().data());
                return Mod::on_initialize();
            }

            const auto real_awesome_function = utility::calculate_absolute(*very_awesome_call + 1);

            if (real_awesome_function == 0) {
                spdlog::error("[{:s}]: Could not find real_awesome_function!", get_name().data());
                return Mod::on_initialize();
            }

            spdlog::info("[{:s}]: Patching very awesome method!", get_name().data());

            very_cool_patches.emplace_back(Patch::create(real_awesome_function, { 0xB0, 0x00, 0xC3 }, true));
        } else {
            spdlog::error("[{:s}]: Could not find very_awesome_method!", get_name().data());
        }
    } else {
        spdlog::error("[{:s}]: Could not find very_awesome_type!", get_name().data());
    }
    }

    s_patch_count_checked = false;

    spdlog::info("Done.");

    return Mod::on_initialize();
}

void IntegrityCheckBypass::on_frame() {
    const auto& gi = sdk::GameIdentity::get();

    re9_heartbeat_bypass();

    {
        static bool s_reported_no_match = false;

        if (!s_reported_no_match && !s_seen_pak_families.empty() && !s_auto_assigned
            && !m_custom_pak_in_directory_paths.empty()) {
            s_reported_no_match = true;

            spdlog::error("[IntegrityCheckBypass]: {} custom pak(s) were cached but NONE were injected - "
                "no archive family matched the auto-assign suffix. Families seen this run:",
                m_custom_pak_in_directory_paths.size());

            for (const auto& family : s_seen_pak_families) {
                spdlog::error("[IntegrityCheckBypass]:   {}", utility::narrow(family));
            }
        }
    }

    if (gi.is_re3()) {
        if (m_bypass_integrity_checks != nullptr) {
            *m_bypass_integrity_checks = true;
        }
    }

    if (gi.is_re8()) {
        // These three are responsible for various stutters and
        // gameplay altering effects e.g. not being able to interact with objects
        disable_update_timers("app.InteractManager");
        disable_update_timers("app.EnemyManager");
        disable_update_timers("app.GUIManager");
        disable_update_timers("app.HIDManager");
        disable_update_timers("app.FadeManager");
    }
}

void IntegrityCheckBypass::disable_update_timers(std::string_view name) const {
    // get the singleton correspdonding to the given name
    auto manager = sdk::get_managed_singleton<::REManagedObject>(name);

    // If the interact manager is null, we're probably not in the game
    if (manager == nullptr || manager->info == nullptr || manager->info->get_class_info() == nullptr) {
        return;
    }

    // Get the sdk::RETypeDefinition of the manager
    auto t = manager->get_type_definition();

    if (t == nullptr) {
        return;
    }

    // Get the update timer fields, which are responsible for disabling interactions (for app.InteractManager)
    // if the integrity checks are triggered
    auto update_timer_enable_field = t->get_field("UpdateTimerEnable");
    auto update_timer_late_enable_field = t->get_field("LateUpdateTimerEnable");

    // Get the actual field data now within the manager
    if (update_timer_enable_field != nullptr) {
        auto& update_timer_enable = update_timer_enable_field->get_data<bool>(manager, true);

        // Log that we are about to set these to false if they were true before
        if (update_timer_enable) {
            spdlog::info("[{:s}]: {:s}.UpdateTimerEnable was true, disabling it...", get_name().data(), name.data());
        }

        update_timer_enable = false;
    }

    if (update_timer_late_enable_field != nullptr) {
        auto& update_timer_late_enable = update_timer_late_enable_field->get_data<bool>(manager, true);

        if (update_timer_late_enable) {
            spdlog::info("[{:s}]: {:s}.LateUpdateTimerEnable was true, disabling it...", get_name().data(), name.data());
        }

        update_timer_late_enable = false;
    }
}

void IntegrityCheckBypass::ignore_application_entries() {
    Hooks::get()->ignore_application_entry(0x76b8100bec7c12c3);
    Hooks::get()->ignore_application_entry(0x9f63c0fc4eea6626);

    const auto& gi = sdk::GameIdentity::get();
    if (gi.tdb_ver() >= 73) {
        Hooks::get()->ignore_application_entry(0x00c0ab9309584734);
        Hooks::get()->ignore_application_entry(0xa474f1d3a294e6a4);
    }
    if (gi.tdb_ver() >= 74) {
        Hooks::get()->ignore_application_entry(0x00ec4793097cd833);
        Hooks::get()->ignore_application_entry(0x00d85893096c4c0c);
    }
}

void IntegrityCheckBypass::immediate_patch_re8() {
    // Apparently patching this in SF6 causes some bugs like chat not showing up and being unable to view replays.
    // Disabling it for now as the game still seems to work fine without it.
    const auto& gi = sdk::GameIdentity::get();
    if (gi.is_sf6()) {
        return;
    }

    if (gi.tdb_ver() < 73) {
    // We have to immediately patch this at startup in RE8 unlike MHRise
    // because the game immediately starts checking the integrity of the executable
    // on the first execution of this callback, unlike MHRise which was delayed.
    // So we can't use the IL2CPP metadata yet, we have to resort to
    // plain old pattern scanning.
    // We're essentially patching the application entries we ignored above, but immediately.
    spdlog::info("[IntegrityCheckBypass]: Scanning RE8...");

    const auto game = utility::get_executable();
    const auto game_size = utility::get_module_size(game).value_or(0);
    const auto game_end = (uintptr_t)game + game_size;

    // Present in MHRise and RE8.
    // sub rax, 128E329h
    const uint32_t sussy_constant = 0x128E329;
    std::optional<uintptr_t> sussy_result{};

    bool patched_sussy1 = false;

    for (sussy_result = utility::scan_data(game, (const uint8_t*)&sussy_constant, sizeof(sussy_constant)); 
         sussy_result.has_value(); 
         sussy_result = utility::scan_data(*sussy_result + 1, (game_end - (*sussy_result + 1)) - 0x100, (const uint8_t*)&sussy_constant, sizeof(sussy_constant)))
    {
        // Find the start of the instruction, given the sussy_constant is in the middle of it.
        const auto resolved_instruction = utility::resolve_instruction(*sussy_result);

        // If this instruction didn't get resolved, go onto the next one. We probably ran into garbage data.
        if (resolved_instruction) {
            const auto sussy_function_start = utility::find_function_start(resolved_instruction->addr);

            if (!sussy_function_start) {
                spdlog::error("[IntegrityCheckBypass]: Could not find function start for sussy_constant @ 0x{:x}", *sussy_result);
                continue;
            }

            // Create a patch that returns instantly.
            static auto patch = Patch::create(sussy_function_start.value(), { 0xC3 }, true);
            patched_sussy1 = true;
            spdlog::info("[IntegrityCheckBypass]: Patched sussy_function 1");
            break;
        }
    }

    if (!patched_sussy1) {
        spdlog::error("[IntegrityCheckBypass]: Could not find sussy_constant usage!");
    }

    // Now we need to patch the second callback.
    // I hope this constant isn't randomly generated by the protection!!!!!!
    /*
        call    ProtectionTripResult
        cmp     eax, 1F2h
        jz      ...
        lea     rcx, ProtectionGlobalContext
        call    ProtectionTripResult
    */
    const auto sussy_result_2 = utility::scan(game, "E8 ? ? ? ? 3D F2 01 00 00 0F 84 ? ? ? ? 48 8D 0D ? ? ? ? E8");

    if (sussy_result_2) {
        const auto sussy_function_start = utility::find_function_start(sussy_result_2.value());

        if (sussy_function_start) {
            static auto patch = Patch::create(sussy_function_start.value(), { 0xC3 }, true);
            spdlog::info("[IntegrityCheckBypass]: Patched sussy_function 2");
        }
    } else {
        spdlog::error("[IntegrityCheckBypass]: Could not find sussy_result_2!");
    }

    // These are embedded checks that run during startup and sometimes during loading transitions
    // they get passed different indices that make it perform different behavior
    // if it returns 1, the original execution flow gets altered
    // and stuff like DLC loading gets skipped so it needs to always return 0
    // there are really obvious constants to go off of within these functions
    // but they look like they might be auto generated so can't rely on them
    const auto sussy_result_3 = utility::scan(game, "8D ? 02 E8 ? ? ? ? 0F B6 C8 48 ? ? 50 48 ? ? 18 0F");

    if (sussy_result_3) {
        const auto func = utility::calculate_absolute(*sussy_result_3 + 4);
        static auto patch = Patch::create(func, { 0xB0, 0x00, 0xC3 }, true);
        spdlog::info("[IntegrityCheckBypass]: Patched sussy_function 3");
    } else {
        const auto sussy_result_alternative = utility::scan(game, "8D ? 05 E8 ? ? ? ? 0F B6 C8 48 ? ? 50 48 ? ? 18 0F");

        if (sussy_result_alternative) {
            const auto func = utility::calculate_absolute(*sussy_result_alternative + 4);
            static auto patch = Patch::create(func, { 0xB0, 0x00, 0xC3 }, true);
            spdlog::info("[IntegrityCheckBypass]: Patched sussy_function 3");
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not find sussy_result_3!");
        }
    }

    const auto sussy_result_4 = utility::scan(game, "72 ? 41 8B ? E8 ? ? ? ? 0F B6 C8 48 ? ? 50 48 ? ? 18 0F");

    if (sussy_result_4) {
        const auto func = utility::calculate_absolute(*sussy_result_4 + 6);
        static auto patch = Patch::create(func, { 0xB0, 0x00, 0xC3 }, true);
        spdlog::info("[IntegrityCheckBypass]: Patched sussy_function 4");
    } else {
        spdlog::error("[IntegrityCheckBypass]: Could not find sussy_result_4!");
    }
    }
}

void IntegrityCheckBypass::immediate_patch_re4() {
    // This patch fixes the constant scans that are done every frame on the game's memory.
    // The scans will still be performed, but the crash will be avoided.
    // Ideally, the scans should be patched as well, because they are literally running every frame
    // which may cause performance issues.
    // As far as I can tell, this conditional jmp jumps into a via::clr::VM cleanup routine, corrupting memory.
    // This will cause the game to crash in a random location that accesses VM memory.
    // This is probably to make finding the code that causes the crash in the first place harder.
    spdlog::info("[IntegrityCheckBypass]: Scanning RE4...");

    const auto game = utility::get_executable();
    const auto conditional_jmp_block = utility::scan(game, "48 8B 8D D0 03 00 00 48 29 C1 75 ?");

    if (!conditional_jmp_block) {
        spdlog::error("[IntegrityCheckBypass]: Could not find conditional_jmp, trying fallback.");

        // mov     [rbp+192h], al
        // this is used shortly after the conditional jmp, only place that uses it.
        const auto unique_instruction = utility::scan(game, "88 85 92 01 00 00");

        if (!unique_instruction) {
            spdlog::error("[IntegrityCheckBypass]: Could not find unique_instruction!");
            return;
        }

        // Conditional jmp is very close to the instruction, before it.
        // However, this specific block of instructions is used all over the place
        // so we have to use the unique instruction is a reference point to scan from.
        const auto short_jmp_before = utility::scan_reverse(*unique_instruction, 0x100, "75 ? 50 F7 D0");

        if (short_jmp_before) {
            static auto patch = Patch::create(*short_jmp_before, { 0xEB }, true);
            spdlog::info("[IntegrityCheckBypass]: Patched conditional_jmp!");
            return;
        }

        // If we've gotten to this point, we are trying the scorched earth method of trying to obtain the function "start"
        // for this giant obfuscated blob. We will get the instructions behind the unique_instruction we found by doing that,
        // and look for the nearest branch instruction to patch.
        spdlog::error("[IntegrityCheckBypass]: Could not find short_jmp_before, trying fallback.");

        // Get the preceding instructions. If this doesn't work we'll need to scan for a common instruction anchor to scan forward from...
        auto previous_instructions = utility::get_disassembly_behind(*unique_instruction);

        if (previous_instructions.empty()) {
            spdlog::error("[IntegrityCheckBypass]: Could not find previous_instructions!");
            return;
        }

        // Reverse the order of the instructions.
        std::reverse(previous_instructions.begin(), previous_instructions.end());

        spdlog::info("[IntegrityCheckBypass]: Found {} previous instructions.", previous_instructions.size());
        spdlog::info("[IntegrityCheckBypass]: Walking previous instructions...");

        for (auto& insn : previous_instructions) {
            if (insn.instrux.BranchInfo.IsBranch) {
                spdlog::info("[IntegrityCheckBypass]: Found branch instruction, patching...");
                
                if (insn.instrux.BranchInfo.IsFar) {
                    static auto patch = Patch::create(insn.addr, { 0xE9 }, true);
                } else {
                    static auto patch = Patch::create(insn.addr, { 0xEB }, true);
                }

                spdlog::info("[IntegrityCheckBypass]: Patched conditional_jmp");
                return;
            }
        }
        
        spdlog::error("[IntegrityCheckBypass]: Could not find branch instruction to patch!");
        return;
    }

    const auto conditional_jmp = *conditional_jmp_block + 10;

    // Create a patch that always jumps.
    static auto patch = Patch::create(conditional_jmp, { 0xEB }, true);

    spdlog::info("[IntegrityCheckBypass]: Patched conditional_jmp!");
}

void* IntegrityCheckBypass::renderer_create_blas_hook(void* a1, void* a2, void* a3, void* a4, void* a5) {
    if (s_corruption_when_zero != nullptr) {
        if (*s_corruption_when_zero == 0) {
            *s_corruption_when_zero = s_last_non_zero_corruption;
            spdlog::info("[IntegrityCheckBypass]: Fixed corruption_when_zero!");
        }

        s_last_non_zero_corruption = *s_corruption_when_zero;
    }

    return s_renderer_create_blas_hook->get_original<decltype(renderer_create_blas_hook)>()(a1, a2, a3, a4, a5);
}

// This is used to nuke the heap allocated code that causes crashes
// when debuggers are attached and other integrity checks.
// They happen to be in the same (heap allocated) executable section, so we can just
// replace every byte with a RET instruction.
void IntegrityCheckBypass::nuke_heap_allocated_code(uintptr_t addr) {
    // Get the base of the memory region.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery((LPCVOID)addr, &mbi, sizeof(mbi)) == 0) {
        spdlog::error("[IntegrityCheckBypass]: VirtualQuery failed!");
        return;
    }
    
    // Get the end of the memory region.
    const auto start = (uintptr_t)mbi.BaseAddress;
    const auto end = (uintptr_t)mbi.BaseAddress + mbi.RegionSize;

    spdlog::info("[IntegrityCheckBypass]: Nuking heap allocated code at 0x{:X} - 0x{:X}", start, end);

    // Fix the protection of the memory region.
    ProtectionOverride _{(void*)start, mbi.RegionSize, PAGE_EXECUTE_READWRITE};

    // Replace every single byte with a RET (C3) instruction.
    std::memset((void*)start, 0xC3, mbi.RegionSize);

    spdlog::info("[IntegrityCheckBypass]: Nuked heap allocated code at 0x{:X}", start);
}

void IntegrityCheckBypass::anti_debug_watcher() try {
    static const auto ntdll = GetModuleHandleW(L"ntdll.dll");
    static const auto dbg_ui_remote_breakin = ntdll != nullptr ? GetProcAddress(ntdll, "DbgUiRemoteBreakin") : nullptr;
    static auto original_dbg_ui_remote_breakin_bytes = dbg_ui_remote_breakin != nullptr ? utility::get_original_bytes(dbg_ui_remote_breakin) : std::optional<std::vector<uint8_t>>{};

    if (dbg_ui_remote_breakin == nullptr) {
        return;
    }

    // We can generally assume it's not hooked at this point if the original bytes are empty.
    if (!original_dbg_ui_remote_breakin_bytes || original_dbg_ui_remote_breakin_bytes->empty()) {
        spdlog::info("[IntegrityCheckBypass]: Manually copying original bytes for DbgUiRemoteBreakin.");
        if (!original_dbg_ui_remote_breakin_bytes) {
            original_dbg_ui_remote_breakin_bytes = std::vector<uint8_t>{};
        }
    }

    if (original_dbg_ui_remote_breakin_bytes->size() < 32) {
        std::copy_n((uint8_t*)dbg_ui_remote_breakin + original_dbg_ui_remote_breakin_bytes->size(), 32 - original_dbg_ui_remote_breakin_bytes->size(), std::back_inserter(*original_dbg_ui_remote_breakin_bytes));
    }

    const uint64_t* first_8_bytes = (uint64_t*)dbg_ui_remote_breakin;
    const uint8_t* first_8_bytes_ptr = (uint8_t*)dbg_ui_remote_breakin;

    if (*(uint64_t*)original_dbg_ui_remote_breakin_bytes->data() != *first_8_bytes) {
        spdlog::info("[IntegrityCheckBypass]: DbgUiRemoteBreakin was hooked, restoring original bytes.");

        if (first_8_bytes_ptr[0] == 0xE9) {
            spdlog::info("[IntegrityCheckBypass]: DbgUiRemoteBreakin was directly hooked, resolving...");
            const auto resolved_jmp = utility::calculate_absolute((uintptr_t)dbg_ui_remote_breakin + 1);
            const auto is_heap_allocated = utility::get_module_within(resolved_jmp).value_or(nullptr) == nullptr;

            if (is_heap_allocated && !IsBadReadPtr((void*)resolved_jmp, 32)) {
                spdlog::info("[IntegrityCheckBypass]: Nuking heap allocated code at 0x{:X}", resolved_jmp);
                nuke_heap_allocated_code(resolved_jmp);
            }
        } else if (first_8_bytes_ptr[0] == 0xFF && first_8_bytes_ptr[1] == 0x25) {
            spdlog::info("[IntegrityCheckBypass]: DbgUiRemoteBreakin was indirectly hooked, resolving...");
            const auto resolved_ptr = utility::calculate_absolute((uintptr_t)dbg_ui_remote_breakin + 2);
            const auto resolved_jmp = *(uintptr_t*)resolved_ptr;
            const auto is_heap_allocated = utility::get_module_within(resolved_jmp).value_or(nullptr) == nullptr;

            if (is_heap_allocated && !IsBadReadPtr((void*)resolved_jmp, 32)) {
                spdlog::info("[IntegrityCheckBypass]: Nuking heap allocated code at 0x{:X}", resolved_jmp);
                nuke_heap_allocated_code(resolved_jmp);
            }
        }
        
        ProtectionOverride _{dbg_ui_remote_breakin, original_dbg_ui_remote_breakin_bytes->size(), PAGE_EXECUTE_READWRITE};
        std::copy(original_dbg_ui_remote_breakin_bytes->begin(), original_dbg_ui_remote_breakin_bytes->end(), (uint8_t*)dbg_ui_remote_breakin);

        spdlog::info("[IntegrityCheckBypass]: Restored DbgUiRemoteBreakin.");
    }
} catch (const std::exception& e) {
    spdlog::error("[IntegrityCheckBypass]: Exception in anti_debug_watcher: {}", e.what());
} catch (...) {
    spdlog::error("[IntegrityCheckBypass]: Unknown exception in anti_debug_watcher!");
}

void IntegrityCheckBypass::init_anti_debug_watcher() {
    if (s_anti_anti_debug_thread != nullptr) {
        return;
    }

    // Run the original watcher once so we get it at least without creating a thread first.
    anti_debug_watcher();

    s_anti_anti_debug_thread = std::make_unique<std::jthread>([](std::stop_token stop_token) {
        spdlog::info("[Thread] anti_debug_watcher TID {}", GetCurrentThreadId());
        spdlog::info("[IntegrityCheckBypass]: Hello from anti_debug_watcher!");
        spdlog::info("[IntegrityCheckBypass]: Waiting for REFramework startup to finish...");

        while (g_framework == nullptr) {
            std::this_thread::yield();
        }

        spdlog::info("[IntegrityCheckBypass]: REFramework startup finished!");

        while (!stop_token.stop_requested()) {
            anti_debug_watcher();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
}

std::optional<uint16_t> get_pak_flags(const std::filesystem::path& path) try {
    std::ifstream f{path, std::ios::binary};

    if (!f) {
        return std::nullopt;
    }

    std::array<uint8_t, 8> header{};

    if (!f.read(reinterpret_cast<char*>(header.data()), header.size())) {
        return std::nullopt;
    }

    if (std::memcmp(header.data(), "KPKA", 4) != 0) {
        return std::nullopt;
    }

    uint16_t flags{};
    std::memcpy(&flags, header.data() + 6, sizeof(flags));

    return flags;
} catch (const std::exception& e) {
    spdlog::error("[IntegrityCheckBypass]: Exception in get_pak_flags: {}", e.what());
    return std::nullopt;
} catch (...) {
    spdlog::error("[IntegrityCheckBypass]: Unknown exception in get_pak_flags!");
    return std::nullopt;
}

#pragma region PAK_LOADING

bool IntegrityCheckBypass::pak_load_check_function(void* pak_struct, const wchar_t* pak_name_wstr, uintptr_t a3, uintptr_t is_mount, uintptr_t a5, uintptr_t a6, uintptr_t a7) {
    const auto return_address = (uintptr_t)_ReturnAddress();

    spdlog::info("[IntegrityCheckBypass]: pak_load_check_function called from: 0x{:X}", return_address);

    if (pak_name_wstr == nullptr) {
        spdlog::warn("[IntegrityCheckBypass]: pak_name_wstr is nullptr!");
        return s_pak_load_check_function_hook.call<bool>(pak_struct, pak_name_wstr, a3, is_mount, a5, a6, a7);
    }

    spdlog::info("[IntegrityCheckBypass]: Pak name: {}", utility::narrow(pak_name_wstr));
    spdlog::info("[IntegrityCheckBypass]: PakLoad entry");
    spdlog::info("  pak_struct: 0x{:X}", reinterpret_cast<uintptr_t>(pak_struct));
    spdlog::info("  pak_name_wstr: {}", utility::narrow(pak_name_wstr));
    spdlog::info("  a3: 0x{:X}", a3);
    spdlog::info("  is_mount: 0x{:X}", is_mount);
    spdlog::info("  a5: 0x{:X}", a5);
    spdlog::info("  a6: 0x{:X}", a6);
    spdlog::info("  a7: 0x{:X}", a7);

    const auto exe = utility::get_executable();
    const auto exe_path = utility::get_module_pathw(exe);

    if (!exe_path) {
        return s_pak_load_check_function_hook.call<bool>(pak_struct, pak_name_wstr, a3, is_mount, a5, a6, a7);;
    }

    std::filesystem::path pak_path{pak_name_wstr};

    if (pak_path.filename() == L"re_chunk_000.pak") {
        memcpy(s_pristine_pak_struct.data(), pak_struct, s_pristine_pak_struct.size());

        //spdlog::info("[IntegrityCheckBypass]: Found pak_ctor at 0x{:X}", (uintptr_t)pak_ctor);

        // First find where pak_struct is pointed to on the stack next to a bunch of nullptrs.
        // This means that's the start of the pak array.
        auto stack = reinterpret_cast<uintptr_t*>(_AddressOfReturnAddress()); // Approximation of stack start

        for (int i = 0; i < 0x5000; ++i) {
            if (stack[i] == reinterpret_cast<uintptr_t>(pak_struct)) {
                size_t num_nullptrs = 0;

                for (size_t j = i + 1; j < i + 1000; ++j) {
                    if (stack[j] == 0) {
                        ++num_nullptrs;
                    } else {
                        break;
                    }
                }

                // The pak array is 505 elements long (at time of writing; automatically determined via num_nullptrs)
                // So if we find 100 nullptrs after the pak_struct, we can assume this is the start of the array.
                // It's guaranteed to be memset to 0, bar the first one which is our struct for re_chunk_000.pak.
                if (num_nullptrs >= 100) {
                    spdlog::info("[IntegrityCheckBypass]: Found pak_struct on stack at index {} with {} nullptrs after it.", i, num_nullptrs);
                    s_pak_array_start = &stack[i];
                    s_pak_array_len = num_nullptrs + 1;
                    break;
                }
            }
        }

        // Sweep the pristine object looking for a CreateEvent-ish looking handle;
        // Skipping vtable.
        for (size_t i = sizeof(void*); i < std::min(s_pristine_pak_struct.size(), static_cast<size_t>(0x200)); i += sizeof(void*)) {
            const auto ptr = *reinterpret_cast<uintptr_t*>(s_pristine_pak_struct.data() + i);

            if (ptr == 0 || ptr == (uintptr_t)INVALID_HANDLE_VALUE || ptr == 0xFFFFFFFF) {
                continue;
            }

            // If it lies within a module, it's not a handle
            if (utility::get_module_within((uintptr_t)ptr).has_value()) {
                continue;
            }

            DWORD handle_flags = 0;
            if (GetHandleInformation((HANDLE)ptr, &handle_flags)) {
                if (s_event_handle_offset == 0) {
                    s_event_handle_offset = i;
                    spdlog::info("[IntegrityCheckBypass]: Found event handle at offset 0x{:X}", s_event_handle_offset);
                } else if (s_event_handle_offset_2 == 0) {
                    s_event_handle_offset_2 = i;
                    spdlog::info("[IntegrityCheckBypass]: Found second event handle at offset 0x{:X}", s_event_handle_offset_2);
                    break;
                }
            }
        }

        // Detect self-referential pointers in the pak struct
        for (size_t i = 0; i < s_pristine_pak_struct.size(); i += sizeof(void*)) {
            auto v = *reinterpret_cast<uintptr_t*>((uintptr_t)pak_struct + i);

            // interior/self-referential pointer -> rebase into the clone
            if (v >= (uintptr_t)pak_struct && v < (uintptr_t)pak_struct + s_pristine_pak_struct.size()) {
                s_pak_rebase_offsets.emplace_back(PakRebase{ .offset = i, .delta_from_base = v - (uintptr_t)pak_struct });
                spdlog::info("[IntegrityCheckBypass]: Detected self-referential pointer at offset 0x{:X} (points to 0x{:X})", i, v);
                continue;
            }
        }
    }

    // Injected paks are opened under their native-form name, so resolve flags from the real file.
    auto flags_path = pak_path;

    if (const auto it = s_injected_name_to_real_path.find(pak_name_wstr); it != s_injected_name_to_real_path.end()) {
        flags_path = it->second;
    }

    if (auto flags = get_pak_flags(flags_path)) {
        s_pak_flags_value = static_cast<uint8_t>(*flags);

        spdlog::info(
            "[IntegrityCheckBypass]: {} flags from disk: 0x{:X}",
            utility::narrow(pak_name_wstr),
            *flags
        );
    } else {
        // Leaving the stale value in place would hand the midhook another pak's flags.
        spdlog::warn("[IntegrityCheckBypass]: Could not read flags from {} (resolved: {})!",
            utility::narrow(pak_name_wstr), utility::narrow(flags_path.wstring()));
    }

    auto res = s_pak_load_check_function_hook.call<bool>(pak_struct, pak_name_wstr, a3, is_mount, a5, a6, a7);
    
    if (res) {
        spdlog::info("[IntegrityCheckBypass]: {} +480 = 0x{:X}", pak_path.filename().string(), *(uint32_t*)((uintptr_t)pak_struct + 0x1E0));
    } else {
        spdlog::warn("[IntegrityCheckBypass]: pak_load_check_function_hook returned false for string: {}", utility::narrow(pak_name_wstr));
    }

    return res;
}

void* IntegrityCheckBypass::pak_load_patch_load_function(uintptr_t* pak_slots, const wchar_t* base_path, int32_t first_slot_index, int32_t load_flags) {
    spdlog::info("[IntegrityCheckBypass]: pak_load_patch_load_function called with base_path: {}", utility::narrow(base_path));

    auto res = IntegrityCheckBypass::s_pak_load_patch_load_hook.call<void*>(pak_slots, base_path, first_slot_index, load_flags);
    auto* slots = reinterpret_cast<uintptr_t*>(*pak_slots);

    if (slots == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: slot array holder was empty, cannot inject custom paks!");
        return res;
    }

    static std::unordered_set<std::wstring> s_injected_families{};

    if (s_injected_families.contains(base_path)) {
        spdlog::info("[IntegrityCheckBypass]: family '{}' already injected, skipping.", utility::narrow(base_path));
        return res;
    }

    size_t num_paks = 0;
    int next_free_patch_index = 1;

    while (next_free_patch_index < 32 && slots[first_slot_index + next_free_patch_index] != 0) {
        ++next_free_patch_index;
    }

    s_seen_pak_families.emplace(base_path);

    bool did_auto_assign = false;

    for (auto str : IntegrityCheckBypass::get_shared_instance()->m_custom_pak_in_directory_paths) {
    //for (auto str : std::array<std::wstring, 2>{L"re_chunk_001.pak", L"re_chunk_002.pak"}) {
        //auto fake_pak = VirtualAlloc(nullptr, s_pristine_pak_struct.size(), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        const auto mod_filename = std::filesystem::path(str).filename().wstring();
        const auto patch_marker = mod_filename.rfind(L".patch_");

        int patch_index = 0;
        bool explicit_target = false;

        if (patch_marker != std::wstring::npos && mod_filename.compare(0, patch_marker, base_path) == 0) {
            int parsed_patch_num = 0;
            size_t digit_pos = patch_marker + 7; // past ".patch_"
            const size_t first_digit = digit_pos;

            while (digit_pos < mod_filename.size() && iswdigit(mod_filename[digit_pos])) {
                parsed_patch_num = parsed_patch_num * 10 + (mod_filename[digit_pos] - L'0');
                ++digit_pos;
            }

            if (digit_pos != first_digit) {
                patch_index = parsed_patch_num;
                explicit_target = true;
            }
        }

        if (!explicit_target) {
            static constexpr std::wstring_view AUTO_ASSIGN_FAMILY_SUFFIX = L".sub_000.pak";

            if (s_auto_assigned || !std::wstring_view{base_path}.ends_with(AUTO_ASSIGN_FAMILY_SUFFIX)) {
                continue;
            }

            patch_index = next_free_patch_index++;
            did_auto_assign = true;
        }

        // Never overwrite an occupied slot: that pointer is a pak the engine loaded, and clobbering
        // it would drop a real archive out of the staging array before registration.
        if (slots[first_slot_index + patch_index] != 0) {
            spdlog::error("[IntegrityCheckBypass]: slot {} for '{}' is already occupied by a real pak, skipping.",
                first_slot_index + patch_index, utility::narrow(mod_filename));
            continue;
        }

        wchar_t native_name_buf[1024]{};
        swprintf_s(native_name_buf, L"%ls.patch_%03d.pak", base_path, patch_index);
        const std::wstring native_name_str{native_name_buf};
        const wchar_t* native_name = native_name_str.c_str();

        auto fake_pak = sdk::memory::allocate(0x300);
        bool use_virtual_alloc = false;
        if (fake_pak == nullptr) {
            spdlog::warn("[IntegrityCheckBypass]: Attempting to allocate using VirtualAlloc, as sdk::memory::allocate failed.");
            fake_pak = VirtualAlloc(nullptr, 0x300, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            use_virtual_alloc = true;
        }
        memcpy(fake_pak, s_pristine_pak_struct.data(), s_pristine_pak_struct.size()); // Struct is pristine post-ctor. Removes need to know ctor addr.

        if (s_event_handle_offset != 0) {
            *reinterpret_cast<HANDLE*>((uintptr_t)fake_pak + s_event_handle_offset) = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        }

        if (s_event_handle_offset_2 != 0) {
            *reinterpret_cast<HANDLE*>((uintptr_t)fake_pak + s_event_handle_offset_2) = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        }

        const auto fake_base = (uintptr_t)fake_pak;

        // Fix up any self-referential pointers in the pak_struct, so that they point to the new fake_pak instead of the original pak_struct.
        for (auto& rebase : s_pak_rebase_offsets) {
            auto v = reinterpret_cast<uintptr_t*>(fake_base + rebase.offset);
            *v = (uintptr_t)fake_base + rebase.delta_from_base; // rebase to the new fake_pak
            spdlog::info("[IntegrityCheckBypass]: Rebased pointer at offset 0x{:X} from 0x{:X} to 0x{:X}", rebase.offset, *v, (uintptr_t)fake_base + rebase.delta_from_base);
        }

        // Register BEFORE the load: both the flags read and the file open happen inside PakLoad.
        s_injected_name_to_real_path[native_name_str] = str;

        spdlog::info("[IntegrityCheckBypass]: injecting {} as native-form '{}' (a6={}, slot={}, {})",
            utility::narrow(str), utility::narrow(native_name_str), patch_index,
            first_slot_index + patch_index, explicit_target ? "named target" : "auto-assigned");

        const auto pak_array_i = first_slot_index + patch_index;
        ++num_paks;

        slots[pak_array_i] = reinterpret_cast<uintptr_t>(fake_pak);

        // Force: is_mount = true.
        if (!pak_load_check_function(reinterpret_cast<void*>(slots[pak_array_i]), native_name, 0, 1, 0, patch_index, 0)) {
            --num_paks;
            slots[pak_array_i] = 0; // engine clears the slot on failure before destroying

            if (use_virtual_alloc) {
                VirtualFree(fake_pak, 0, MEM_RELEASE);
            } else {
                sdk::memory::deallocate(fake_pak);
            }

            spdlog::warn("[IntegrityCheckBypass]: pak_load_check_function_hook returned false for string: {}", utility::narrow(str));
        } else {
            spdlog::info("[IntegrityCheckBypass]: inserted pak_struct for string: {} at index {}", utility::narrow(str), pak_array_i);

            /*
            pak + 0x08:  wchar_t buf OR wchar_t* ptr   (SSO: <= 0xB chars inline; else heap ptr at +0x08)
            pak + 0x18:  uint32 size    (char count, w/o null)
            pak + 0x1C:  uint32 capacity (if >= 0xC → buffer is heap pointer at +0x08)
            */

            /*auto fake_str = sdk::memory::allocate((str.size() + 1) * sizeof(wchar_t));
            // We only want the filename part
            const auto filename = std::filesystem::path(str).filename().wstring();
            std::memcpy(fake_str, filename.data(), (filename.size() + 1) * sizeof(wchar_t));
            // log what's at fake_pak + 0x08, +0x18, +0x1C
            spdlog::info("[IntegrityCheckBypass]: fake_pak +0x08 = {}", utility::narrow(*reinterpret_cast<wchar_t**>((uintptr_t)fake_pak + 0x08)));
            spdlog::info("[IntegrityCheckBypass]: fake_pak +0x18 = 0x{:X}", *reinterpret_cast<uint32_t*>((uintptr_t)fake_pak + 0x18));
            spdlog::info("[IntegrityCheckBypass]: fake_pak +0x1C = 0x{:X}", *reinterpret_cast<uint32_t*>((uintptr_t)fake_pak + 0x1C));
            *reinterpret_cast<uintptr_t*>((uintptr_t)fake_pak + 0x08) = reinterpret_cast<uintptr_t>(fake_str);
            *reinterpret_cast<uint32_t*>((uintptr_t)fake_pak + 0x18) = static_cast<uint32_t>(filename.size());
            *reinterpret_cast<uint32_t*>((uintptr_t)fake_pak + 0x1C) = static_cast<uint32_t>(filename.size());*/
        }
    }

    // Latch only if something actually landed, so a family whose loads all failed can retry.
    if (num_paks > 0) {
        s_injected_families.insert(base_path);

        if (did_auto_assign) {
            s_auto_assigned = true;
        }

        spdlog::info("[IntegrityCheckBypass]: injected {} custom pak(s) into family '{}'{}.",
            num_paks, utility::narrow(base_path), did_auto_assign ? " (auto-assigned; no other family will get them)" : "");
    }

    return res;
}

// This allows unencrypted paks to load.
void IntegrityCheckBypass::sha3_rsa_code_midhook(safetyhook::Context& context) {
    spdlog::info("[IntegrityCheckBypass]: sha3_code_midhook called!");
    // Log registers
    spdlog::info("[IntegrityCheckBypass]: RAX: 0x{:X}", context.rax);
    spdlog::info("[IntegrityCheckBypass]: RCX: 0x{:X}", context.rcx);
    spdlog::info("[IntegrityCheckBypass]: RDX: 0x{:X}", context.rdx);
    spdlog::info("[IntegrityCheckBypass]: R8: 0x{:X}", context.r8);
    spdlog::info("[IntegrityCheckBypass]: R9: 0x{:X}", context.r9);
    spdlog::info("[IntegrityCheckBypass]: R10: 0x{:X}", context.r10);
    spdlog::info("[IntegrityCheckBypass]: R11: 0x{:X}", context.r11);
    spdlog::info("[IntegrityCheckBypass]: R12: 0x{:X}", context.r12);
    spdlog::info("[IntegrityCheckBypass]: R13: 0x{:X}", context.r13);
    spdlog::info("[IntegrityCheckBypass]: R14: 0x{:X}", context.r14);
    spdlog::info("[IntegrityCheckBypass]: R15: 0x{:X}", context.r15);
    spdlog::info("[IntegrityCheckBypass]: RSP: 0x{:X}", context.rsp);
    spdlog::info("[IntegrityCheckBypass]: RIP: 0x{:X}", context.rip);
    spdlog::info("[IntegrityCheckBypass]: RBP: 0x{:X}", context.rbp);
    spdlog::info("[IntegrityCheckBypass]: RSI: 0x{:X}", context.rsi);
    spdlog::info("[IntegrityCheckBypass]: RDI: 0x{:X}", context.rdi);

    enum PakFlags : uint8_t {
        ENCRYPTED = 0x8
    };

    //const auto pak_flags = (PakFlags)context.rax; // Might change, maybe add automated register detection later
    PakFlags pak_flags{};

    if (s_pak_flags_value) {
        pak_flags = static_cast<PakFlags>(*s_pak_flags_value);
        spdlog::info("[IntegrityCheckBypass]: Using stored pak flags value: 0x{:X}", *s_pak_flags_value);
    } else {
        spdlog::warn("[IntegrityCheckBypass]: No stored pak flags value...");
    }

    if ((pak_flags & PakFlags::ENCRYPTED) != 0) {
        spdlog::info("[IntegrityCheckBypass]: Pak is encrypted, allowing decryption code to run.");
        return;
    }

    context.rip = *s_sha3_code_end;

    spdlog::info("[IntegrityCheckBypass]: Unencrypted pak detected, skipping decryption code!");
}

void IntegrityCheckBypass::restore_unencrypted_paks() {
    spdlog::info("[IntegrityCheckBypass]: Restoring unencrypted paks...");

    scan_patch_files_count();

    // If this breaks... we'll fix it!
    const auto game = utility::get_executable();
    const auto pak_load_fn = utility::find_function_from_string_ref(game, L"_chunk_", true);
    
    std::optional<uintptr_t> sha3_code_start{};

    // The usual path we'll use. Easily identifies the func via string ref.
    // looks for a basic block containing a bunch of vmovups instructions
    // set the sha3_code_start to that block.
    if (pak_load_fn) {
        spdlog::info("[IntegrityCheckBypass]: Found pak_load_fn @ 0x{:X}, using it as reference to find sha3_code_start!", *pak_load_fn);
        const auto bounds = utility::determine_function_bounds(*pak_load_fn);

        if (bounds) {
            const auto blocks = utility::collect_linear_blocks(bounds->start, bounds->end);
            const utility::LinearBlock* found_block = nullptr;

            for (const auto& block : blocks) {
                // Look for sequences of vmovups instructions using disasm.
                size_t vmovups_sequence_length = 0;
                utility::linear_decode((uint8_t*)block.start, 100, [&](utility::ExhaustionContext& ctx) -> bool {
                    if (ctx.addr > block.end) {
                        return false;
                    }

                    if (std::string_view{ctx.instrux.Mnemonic}.starts_with("VMOVUPS")) {
                        vmovups_sequence_length++;
                    } else {
                        if (vmovups_sequence_length >= 4) { // The decryption code has a long sequence of vmovups instructions, so we look for sequences of 4 or more.
                            spdlog::info("[IntegrityCheckBypass]: Found vmovups sequence of length {} at 0x{:X}, likely sha3_code_start!", vmovups_sequence_length, ctx.addr);
                            found_block = &block;
                            return false;
                        }

                        vmovups_sequence_length = 0;
                    }

                    return true;
                });

                if (found_block != nullptr) {
                    break;
                }
            }

            if (found_block) {
                // The start of the vmovups isn't always the correct spot. The start of the block is actually the correct spot.
                sha3_code_start = found_block->start;
                spdlog::info("[IntegrityCheckBypass]: Found sha3_code_start @ 0x{:X} using vmovups sequence!", *sha3_code_start);
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find vmovups sequence in blocks of pak_load_fn, cannot find sha3_code_start!");
            }
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not determine function bounds for pak_load_fn, cannot find sha3_code_start!");
        }
    } 
    
    // Fall back to old stuff.
    if (!sha3_code_start) {
        std::vector<std::string> possible_patterns = {
            "C5 F8 57 C0 C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 44 24 ? 48", 
            "C5 F8 57 C0 C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? 48 C1 ? 10",  // MHWILDS v1.041
            "C5 F8 57 C0 C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? C5 FC 11 84 24 ? ? ? ? 48 8B ? ? 00 00 00 48 C1 ? 10",  // MHSTORIES3
            "48 8B 05 ? ? ? ? 49 33 ? C0 00 00 00 C5 F1 EF C9 C5 F9 EF C0 C5 FC 11 45 ? C5 FC 11 4D ? C5 FC 11 4D ? C5 FC 11 4D ? C5 FC 11 4D ? 48 A9 00 00 F8 FF",  // PRAGMATA
            "C5 F8 57 C0 C5 FC 11 45 ? C5 FC 11 45 ? C5 FC 11 45 ? C5 FC 11 45 ? C5 FC 11 45 ? 48 C1 E9 10" // RE9 v1.0.0.0
        };

        for (const auto& pattern : possible_patterns) {
            sha3_code_start = utility::scan(game, pattern);
            if (sha3_code_start) {
                break;
            }
        }
    }

    if (!sha3_code_start) {
        spdlog::error("[IntegrityCheckBypass]: Could not find sha3_rsa_code_start!");
        return;
    }
    
    spdlog::info("[IntegrityCheckBypass]: Found sha3_rsa_code_start @ 0x{:X}", *sha3_code_start);

    std::vector<std::string> possible_end_patterns = {
        "48 8B 8E C0 00 00 00 48 C1 E9 ?",
        "48 8B ? C0 00 00 00 48 C1 ? 10 4C 21 ? 48 8B 0D ? ? ? ? 48 C1 ? 10 4C 21 ? 48 39 ? 75 ? 48 83 ? 30 FF 74 ? 31 ? 4C 89 ? 31 ? 45 31 ? C5 F8 77", // MHSTORIES3, hope its the last thing that is like this
        "48 8B 05 ? ? ? ? 49 33 86 C0 00 00 00 48 A9 00 00 F8 FF 75 ? 49 83 7E 30 FF 74 ? 49 8D 4E 30 45 33 C0 33 D2 C5 F8 77",   // PRAGMATA
        "E9 ? ? ? ? 8B 8E 70 01 00 00 48 85 C9 74 ? 48 8B 86 68 01 00 00 48 C1 E1 04", // DD2 dark arisen
    };

    for (const auto& pattern : possible_end_patterns) {
        s_sha3_code_end = utility::scan(game, pattern);
        if (s_sha3_code_end) {
            break;
        }
    }

    if (!s_sha3_code_end) {
        spdlog::error("[IntegrityCheckBypass]: Could not find sha3_rsa_code_end, cannot restore unencrypted paks!");
        return;
    }
    
    spdlog::info("[IntegrityCheckBypass]: Found sha3_rsa_code_end @ 0x{:X}", *s_sha3_code_end);

    s_sha3_rsa_code_midhook = safetyhook::create_mid((void*)*sha3_code_start, &sha3_rsa_code_midhook);

    spdlog::info("[IntegrityCheckBypass]: Created sha3_rsa_code_midhook!");

    const auto& gi = sdk::GameIdentity::get();

    // Enable the pak loading directory pipeline
    if (gi.tdb_ver() >= 81) {
        // Find function start may
        auto pak_load_check_start = utility::find_function_start_unwind(*pak_load_fn);

        if (pak_load_check_start) {
            spdlog::info("[IntegrityCheckBypass]: Found pak_load_check_function @ 0x{:X}, hook!", (uintptr_t)*pak_load_check_start);
            s_pak_load_check_function_hook = safetyhook::create_inline((void*)*pak_load_check_start, &IntegrityCheckBypass::pak_load_check_function);

            find_try_hook_via_file_load_win32_create_file(*pak_load_check_start);
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not find pak_load_check_function start!");
        }

        const wchar_t *patch_version_string = L"/Environment/Package/PatchVersion:";
        const wchar_t *re_chunk_string = L"re_chunk_";
        
        auto load_patch_func = utility::find_function_with_string_refs(game, patch_version_string, re_chunk_string, false, true);
        if (load_patch_func) {
            s_pak_load_patch_load_hook = safetyhook::create_inline((void*)*load_patch_func, &IntegrityCheckBypass::pak_load_patch_load_function);

            if (s_pak_load_patch_load_hook) {
                spdlog::info("[IntegrityCheckBypass]: Found pak_load_patch_load_function @ 0x{:X}, hook!", (uintptr_t)*load_patch_func);
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not create hook for pak_load_patch_load_function!");
            }
        }
    }

    auto previous_instructions = utility::get_disassembly_behind(*s_sha3_code_end);
    auto previous_instructions_start = utility::get_disassembly_behind(*sha3_code_start);

    // This NOPs out the conditional jump that rejects the PAK file if the integrity check fails.
    // Normally, the game computes a SHA3-256 hash of the TOC/resource headers before obfuscation.
    // It then XORs the TOC/resource headers with this hash to make them unreadable.
    // To verify integrity, the game decrypts a precomputed SHA3-256 hash from the PAK header using RSA.
    // If the computed SHA3-256 hash of the TOC/resource headers doesn't match the decrypted one, the PAK is rejected.
    // Without patching, bypassing this check would require obtaining Capcom's private RSA key to sign new hashes,
    // which is infeasible. Instead, we NOP the conditional jump to force the game to accept modded PAKs.
    // (Thanks to Rick Gibbed (@gibbed) for the explanation about how SHA3 and RSA fit together in this context)
    if (!previous_instructions.empty()) {
        const auto& previous_instruction = previous_instructions.back();

        if (previous_instruction.instrux.BranchInfo.IsBranch && previous_instruction.instrux.BranchInfo.IsConditional) {
            spdlog::info("[IntegrityCheckBypass]: Found conditional branch instruction @ 0x{:X}, NOPing it", previous_instruction.addr);

            // NOP out the conditional jump
            std::vector<int16_t> nops{};
            nops.resize(previous_instruction.instrux.Length);
            std::fill(nops.begin(), nops.end(), 0x90);

            static auto patch = Patch::create(previous_instruction.addr, nops, true);

            spdlog::info("[IntegrityCheckBypass]: NOP'd out conditional jump!");
        } else {
            spdlog::warn("[IntegrityCheckBypass]: Previous instruction is not a conditional branch, cannot NOP it!");
        }
    }
}

int IntegrityCheckBypass::scan_patch_files_count() {
    if (s_patch_count_checked) {
        return s_patch_count;
    }

    spdlog::info("[IntegrityCheckBypass]: Scanning for patch files...");

    // Get executable directory
    const auto exe_module = utility::get_executable();
    const auto exe_path = utility::get_module_pathw(exe_module);

    if (!exe_path) {
        spdlog::error("[IntegrityCheckBypass]: Could not get executable path!");
        return -1;
    }

    const auto exe_dir = std::filesystem::path(*exe_path).parent_path();

    spdlog::info("[IntegrityCheckBypass]: Scanning directory: {}", utility::narrow(exe_dir.wstring()));

    // Scan for matching files: re_chunk_000.pak.sub_000.pak.patch_0.pak
    // Capture the patch number to find the highest one
    std::regex pattern(R"(re_chunk_\d+\.pak\.sub_\d+\.pak\.patch_(\d+)\.pak)", std::regex::ECMAScript);
    std::smatch match;
    int highest_patch_num = -1;

    for (const auto& entry : std::filesystem::directory_iterator(exe_dir)) {
        if (entry.is_regular_file()) {
            const auto filename = entry.path().filename().string();
            if (std::regex_match(filename, match, pattern)) {
                try {
                    const int patch_num = std::stoi(match[1].str());
                    highest_patch_num = std::max(highest_patch_num, patch_num);
                    spdlog::info("[IntegrityCheckBypass]: Found patch file: {} (patch_{})", filename, patch_num);
                } catch (const std::exception& e) {
                    spdlog::warn("[IntegrityCheckBypass]: Failed to parse patch number from {}: {}", filename, e.what());
                }
            }
        }
    }

    if (highest_patch_num >= 0) {
        spdlog::info("[IntegrityCheckBypass]: Highest patch number found: {}", highest_patch_num);
    } else {
        spdlog::warn("[IntegrityCheckBypass]: No valid patch files found!");
        highest_patch_num = 0;
    }

    auto integrity_shared_instance = IntegrityCheckBypass::get_shared_instance();
    auto other_custom_paks_count = integrity_shared_instance->cache_and_count_custom_pak_in_directory();

    s_base_directory_patch_count = highest_patch_num;
    highest_patch_num += other_custom_paks_count;

    s_patch_count_checked = true;
    s_patch_count = highest_patch_num;

    return highest_patch_num;
}

void IntegrityCheckBypass::immediate_patch_dd2() {
    // Just like RE4, this deals with the scans that are done every frame on the game's memory.
    // The scans are still performed, but the crash will be avoided.
    // This time, the obfuscation is much worse, and the way the crash is caused is much more indirect.
    // They corrupt something that has something to do with the renderer,
    // possibly with how it updates constant buffers and/or pipeline state
    // this makes the crash look like it comes from DXGI present, due to a GPU error.
    // The place this is happening is very simple, but it was not an easy find due to
    // how indirect it was + all the obfuscation.
    spdlog::info("[IntegrityCheckBypass]: Scanning DD2...");

    const auto game = utility::get_executable();
    const auto game_size = utility::get_module_size(game).value_or(0);
    const auto game_end = (uintptr_t)game + game_size;

    const auto& gi = sdk::GameIdentity::get();
    if (gi.tdb_ver() >= 74) {
    init_anti_debug_watcher();

    // TODO: Check if full release of Pragmata needs this
    // right now it freezes the game
    if (gi.is_mhwilds()) {
            init_debug_break_watcher();
            init_first_chance_exception_logger();
            // MHWilds runs .\CrashReport.exe via ShellExecuteExW and calls ExitProcess(0)
            // (MonsterHunterWilds.exe+0xA4D6E49) unless the child exits with code 0xCD. Do NOT fail
            // ShellExecuteExW: that path sets edi=1 and makes the caller treat the check as failed.
            // Instead force the PathFileExistsW branch to fall through to the "file does not exist"
            // continue path, which returns 0.
            if (const auto crash_report_launch = utility::scan(game, "48 8D 3D ? ? ? ? 48 89 F9 FF 15 ? ? ? ? 85 C0 75 ?")) {
                static auto crash_report_patch = Patch::create(*crash_report_launch + 18, {0x90, 0x90}, true);
                spdlog::info("[IntegrityCheckBypass]: Patched MHWilds CrashReport.exe check (site 1)!");
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find MHWilds CrashReport.exe check (site 1)!");
            }

            init_terminate_process_watcher();
            init_gsfailure_watcher(game);
    const auto query_performance_frequency = &QueryPerformanceFrequency;
    const auto query_performance_counter = &QueryPerformanceCounter;

    if (query_performance_frequency != nullptr && query_performance_counter != nullptr) {
        const auto qpf_import = utility::scan_ptr(game, (uintptr_t)query_performance_frequency);
        const auto qpc_import = utility::scan_ptr(game, (uintptr_t)query_performance_counter);

        if (qpf_import && qpc_import) {
            const auto crasher_fn = utility::find_function_with_refs(game, { *qpf_import, *qpc_import });

            if (crasher_fn) {
                spdlog::info("[IntegrityCheckBypass]: Found crasher_fn!");

                auto crasher_fn_ref = utility::scan_displacement_reference(game, *crasher_fn);

                if (crasher_fn_ref) {
                    spdlog::info("[IntegrityCheckBypass]: Found crasher_fn_ref");
                }

                if (crasher_fn_ref && *(uint8_t*)(*crasher_fn_ref - 1) == 0xE9) {
                    crasher_fn_ref = utility::find_function_start(*crasher_fn_ref - 1);
                } else {
                    crasher_fn_ref = *crasher_fn;
                }

                if (crasher_fn_ref) {
                    spdlog::info("[IntegrityCheckBypass]: Found crasher fn (real)");

                    // We have to use this because I think that the AVX2 scan is broken here for some reason... uh oh...
                    const auto scanner_fn_middle = utility::scan_relative_reference_scalar((uintptr_t)game, game_size - 0x1000, *crasher_fn_ref, [](uintptr_t addr) {
                        return *(uint8_t*)(addr - 1) == 0xE8;
                    });

                    if (scanner_fn_middle) {
                        spdlog::info("[IntegrityCheckBypass]: Found scanner_fn_middle");

                        const auto scanner_fn = utility::find_function_start_unwind(*scanner_fn_middle);

                        if (scanner_fn) {
                            spdlog::info("[IntegrityCheckBypass]: Found scanner_fn!");
                            static auto nuke_patch = Patch::create(*scanner_fn, { 0xC3 }, true); // ret
                            spdlog::info("[IntegrityCheckBypass]: Patched scanner_fn!");
                        } else {
                            spdlog::error("[IntegrityCheckBypass]: Could not find scanner_fn!");
                        }
                    } else {
                        spdlog::error("[IntegrityCheckBypass]: Could not find scanner_fn_middle! (3)");
                    }
                } else {
                    spdlog::error("[IntegrityCheckBypass]: Could not find crasher_fn_ref! (2)");
                }

                // Make function just ret
                //static auto patch = Patch::create(*crasher_fn, { 0xC3 }, true);

                const auto cmp_jz = utility::find_pattern_in_path((uint8_t*)*crasher_fn, 1000, false, "39 0C 82 74 ?");

                if (cmp_jz) {
                    static auto patch = Patch::create(cmp_jz->addr + 3, { 0xEB }, true);
                    spdlog::info("[IntegrityCheckBypass]: Patched crasher_fn!");
                } else {
                    spdlog::error("[IntegrityCheckBypass]: Could not find cmp_jz!");
                }
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find crasher_fn!");
            }
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not find QueryPerformanceFrequency/Counter imports!");
        }
    }
    }

    if (const auto create_blas_fn = utility::find_function_from_string_ref(game, "createBLAS"); create_blas_fn.has_value()) {
        const auto create_blas_fn_unwind = utility::find_function_start_unwind(*create_blas_fn);

        if (create_blas_fn_unwind) {
            spdlog::info("[IntegrityCheckBypass]: Found createBLAS!");

            // Look for first lea rcx, [mem]
            const auto lea_rcx = utility::find_pattern_in_path((uint8_t*)*create_blas_fn_unwind, 1000, false, "48 8D 0D ? ? ? ?");

            if (lea_rcx) {
                s_corruption_when_zero = (uint32_t*)utility::calculate_absolute(lea_rcx->addr + 3);
                spdlog::info("[IntegrityCheckBypass]: Found corruption_when_zero!");
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find lea_rcx!");
            }

            s_renderer_create_blas_hook = std::make_unique<FunctionHook>(*create_blas_fn, &renderer_create_blas_hook);

            if (!s_renderer_create_blas_hook->create()) {
                spdlog::error("[IntegrityCheckBypass]: Failed to hook createBLAS!");
            } else {
                spdlog::info("[IntegrityCheckBypass]: Hooked createBLAS!");
            }
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not find unwound createBLAS!");
        }
    } else {
        spdlog::error("[IntegrityCheckBypass]: Could not find createBLAS!");
    }

    static std::vector<Patch::Ptr> sus_constant_patches{};

    for (auto ref = utility::scan(game, "81 ? E1 53 BD 4C");
         ref.has_value();
         ref = utility::scan(*ref + 1, (game_end - (*ref + 1)) - 0x1000, "81 ? E1 53 BD 4C"))
    {
        // Patch to 0x1337BEEF
        sus_constant_patches.emplace_back(Patch::create(*ref + 2, { 0xEF, 0xBE, 0x37, 0x13 }, true));
    }

    spdlog::info("[IntegrityCheckBypass]: Patched {} sus_constants! (DD2+ variant)", sus_constant_patches.size());

    restore_unencrypted_paks();
    }

    const auto conditional_jmp_block = utility::scan(game, "41 8B ? ? 78 83 ? 07 ? ? 75 ?");

    if (conditional_jmp_block) {
        // Jnz->Jmp
        const auto conditional_jmp = *conditional_jmp_block + 10;

        // Create a patch that always jumps.
        static auto dd2patch = Patch::create(conditional_jmp, { 0xEB }, true);

        spdlog::info("[IntegrityCheckBypass]: Patched conditional_jmp! (DD2)");
    } else {
        spdlog::error("[IntegrityCheckBypass]: Could not find conditional_jmp for DD2, attempting fallback.");

        const auto create_blas_fn = utility::find_function_from_string_ref(game, "createBLAS");

        if (create_blas_fn) {
            const auto and_eax_07_instr = utility::find_pattern_in_path((uint8_t*)*create_blas_fn, 100, false, "83 E0 07");
            
            if (and_eax_07_instr) {
                // Find next conditional jmp and patch it.
                const auto conditional_jmp = utility::scan_mnemonic(and_eax_07_instr->addr + and_eax_07_instr->instrux.Length, 10, "JNZ");

                if (conditional_jmp) {
                    // Jnz->Jmp
                    static auto dd2patch = Patch::create(*conditional_jmp, { 0xEB }, true);

                    spdlog::info("[IntegrityCheckBypass]: Patched conditional_jmp! (DD2)");
                } else {
                    spdlog::error("[IntegrityCheckBypass]: Could not find conditional_jmp for DD2.");
                }
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find and_eax_07 instruction for DD2 fallback.");
            }
        } else {
            spdlog::error("[IntegrityCheckBypass]: Could not find createBLAS!");
        }
    }

    const auto second_conditional_jmp_block = utility::scan(game, "49 3B D0 75 ? ? 8B ? ? ? ? ? ? 8B ? ? ? ? ? ? 8B ? ? 8B ? ? ? ? ?");

    if (second_conditional_jmp_block) {
        // Jnz->Jmp
        const auto second_conditional_jmp = *second_conditional_jmp_block + 3;

        // Create a patch that always jumps.
        static auto dd2patch2 = Patch::create(second_conditional_jmp, { 0xEB }, true);

        spdlog::info("[IntegrityCheckBypass]: Patched second_conditional_jmp! (DD2)");
    } else {
        spdlog::error("[IntegrityCheckBypass]: Could not find second_conditional_jmp for DD2.");
    }

    const auto natives_str_addr = utility::scan(game, "00 00 2F 00 6E 00 61 00 74 00 69 00 76 00 65 00 73 00 2F 00 00 00");

    // the purpose of this is to re-enable loose file loading
    // the game explicitly looks for this string in the path and
    // causes load failures if it finds it
    if (natives_str_addr) {
        spdlog::info("[IntegrityCheckBypass]: Found /natives/ string for DD2. Patching...");

        wchar_t* natives_str = (wchar_t*)(*natives_str_addr + 2);
        DWORD old_protect{};
        VirtualProtect(natives_str, 10 * sizeof(wchar_t), PAGE_EXECUTE_READWRITE, &old_protect);

        spdlog::info("[IntegrityCheckBypass]: /natives/ string: {}", utility::narrow(natives_str));

        // replace string with a completely invalid string that cannot be a valid path
        natives_str[0] = L'?'; // /

        DWORD old2{};
        VirtualProtect(natives_str, 10 * sizeof(wchar_t), old_protect, &old2);

        spdlog::info("[IntegrityCheckBypass]: Patched /natives/ string for DD2.");
    } else {
        spdlog::error("[IntegrityCheckBypass]: Could not find /natives/ string for DD2.");
    }
}

// Temporary workarounds
static SafetyHookInline g_submit_hook{};

static void log_submit_descriptor_once(int64_t descriptor, uintptr_t first_entry, uintptr_t func_ptr) {
    static std::unordered_set<int64_t> seen_descriptors{};
    static std::shared_mutex seen_descriptors_mutex{};

    try {
        // Fast path: check under shared lock (most calls hit this)
        {
            std::shared_lock lock{seen_descriptors_mutex};
            if (seen_descriptors.contains(descriptor)) {
                return;
            }
        }

        // Slow path: take exclusive lock to insert
        std::unique_lock lock{seen_descriptors_mutex};
        if (seen_descriptors.emplace(descriptor).second) {
            SPDLOG_INFO("[IntegrityCheckBypass]: First time seeing descriptor 0x{:X}, Entry: 0x{:X}, Func Ptr: 0x{:X}", descriptor, first_entry, func_ptr);
        }
    } catch (...) {
    }
}

static std::unordered_map<int64_t, uintptr_t>& get_submit_descriptor_original_func_ptrs() {
    static std::unordered_map<int64_t, uintptr_t> original_func_ptrs{};
    return original_func_ptrs;
}

static std::shared_mutex& get_submit_descriptor_original_func_ptrs_mutex() {
    static std::shared_mutex original_func_ptrs_mutex{};
    return original_func_ptrs_mutex;
}

static uintptr_t get_submit_descriptor_original_func_ptr(int64_t descriptor) {
    if (descriptor == 0) {
        return 0;
    }

    try {
        std::shared_lock lock{get_submit_descriptor_original_func_ptrs_mutex()};
        auto& original_func_ptrs = get_submit_descriptor_original_func_ptrs();
        auto it = original_func_ptrs.find(descriptor);
        if (it != original_func_ptrs.end()) {
            return it->second;
        }
    } catch (...) {
    }

    return 0;
}

static void remember_submit_descriptor_original_func_ptr(int64_t descriptor, uintptr_t func_ptr) {
    if (descriptor == 0 || func_ptr == 0) {
        return;
    }

    try {
        if (get_submit_descriptor_original_func_ptr(descriptor) == func_ptr) {
            return; // Only incur cost of a shared mutex.
        }

        std::unique_lock lock{get_submit_descriptor_original_func_ptrs_mutex()};
        get_submit_descriptor_original_func_ptrs()[descriptor] = func_ptr;
    } catch (...) {
    }
}

uintptr_t __fastcall hk_JobQueue_SubmitDescriptor(uintptr_t scheduler, int64_t descriptor, int priority, uint32_t max_workers) {
    auto first_entry = *reinterpret_cast<uintptr_t*>(descriptor + 24);
    uintptr_t func_ptr = 0;

    if (first_entry) {
        func_ptr = *reinterpret_cast<uintptr_t*>(first_entry + 8);

        if (IsBadReadPtr((void*)func_ptr, 8)) {
            // whatever.
            return g_submit_hook.call<uintptr_t>(scheduler, descriptor, priority, max_workers);
        }
    }

    log_submit_descriptor_once(descriptor, first_entry, func_ptr);

    if (first_entry) {
        __try {
            if (func_ptr && *reinterpret_cast<uint16_t*>(func_ptr) == 0x0B0F) {
                const auto original_func_ptr = get_submit_descriptor_original_func_ptr(first_entry);

                if (original_func_ptr != 0 && original_func_ptr != func_ptr) {
                    *reinterpret_cast<uintptr_t*>(first_entry + 8) = original_func_ptr;
                    SPDLOG_INFO("[IntegrityCheckBypass]: Restored descriptor 0x{:X} func pointer to 0x{:X} (was 0x{:X})", descriptor, original_func_ptr, func_ptr);
                }

                auto func_ptr_addr = first_entry + 8;
                SPDLOG_INFO("[IntegrityCheckBypass]: HWBP TARGET: 0x{:X} (entry+8 at 0x{:X})", func_ptr_addr, first_entry);

                const auto retaddr = (uintptr_t)_ReturnAddress();
                SPDLOG_INFO("[IntegrityCheckBypass]: Caught integrity check job submission! Descriptor: 0x{:X}, Func Ptr: 0x{:X}, Return Address: 0x{:X}", descriptor, func_ptr, retaddr);
            }

            if (func_ptr) {
                remember_submit_descriptor_original_func_ptr(first_entry, func_ptr);
            }
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            SPDLOG_WARN("[IntegrityCheckBypass]: Exception caught while checking integrity job submission. Descriptor: 0x{:X}", descriptor);
            return g_submit_hook.call<uintptr_t>(scheduler, descriptor, priority, max_workers);
            //return 0; // garbage pointer - eat it
        }
    }
    return g_submit_hook.call<uintptr_t>(scheduler, descriptor, priority, max_workers);
}

// Harmless replacement - just returns
static void __fastcall noop_job(int64_t, int64_t) {}

template<int reg>
void validate_job_func(SafetyHookContext& ctx) {
    auto func_ptr = ctx.rax;
    if (!func_ptr) {
        return;
    }

    // inline isbadreadptr recreation so we don't call out into kernel32
    __try {
        volatile uint64_t dummy = *(volatile uint64_t*)func_ptr;
        (void)dummy;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return;
    }

    __try {
        // UD2
        if (*reinterpret_cast<uint16_t*>(func_ptr) == 0x0B0F) {
            // if we already have a cached original, restore it to prevent crashes.
            const auto original_func_ptr = get_submit_descriptor_original_func_ptr(disasm_utils::get_register_value(ctx, reg));
            if (original_func_ptr != 0 && original_func_ptr != func_ptr) {
                ctx.rax = original_func_ptr;
                *(uintptr_t*)(disasm_utils::get_register_value(ctx, reg) + 8) = original_func_ptr; // restore the func ptr in the descriptor as well.
                SPDLOG_INFO("[IntegrityCheckBypass]: Restored descriptor 0x{:X} func pointer to 0x{:X} in job func validation (was 0x{:X})", disasm_utils::get_register_value(ctx, reg), original_func_ptr, func_ptr);
            } else {
                ctx.rax = reinterpret_cast<uintptr_t>(&noop_job);
                //SPDLOG_INFO("[IntegrityCheckBypass]: Caught integrity check job submission at call site, skipping! FuncPtr: 0x{:X}", func_ptr);
            }
        } else {
            // also cache the original here for later.
            remember_submit_descriptor_original_func_ptr(disasm_utils::get_register_value(ctx, reg), func_ptr);
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ctx.rax = reinterpret_cast<uintptr_t>(&noop_job);
        SPDLOG_WARN("[IntegrityCheckBypass]: Exception caught while validating job function pointer. FuncPtr: 0x{:X}", func_ptr);
    }
}

void IntegrityCheckBypass::immediate_patch_re9() {
    spdlog::info("[IntegrityCheckBypass]: Scanning RE9...");

    const auto game = utility::get_executable();
    const auto game_size = utility::get_module_size(game).value_or(0);
    const auto game_end = (uintptr_t)game + game_size;

    static std::vector<Patch::Ptr> sus_constant_patches2{};

    // Fixes calls into BushClover. BushClover is a manually mapped DLL in the RE Engine that causes a fake UD2 exception
    // using a manually crafted exception that calls into KiUserExceptionDispatcher, triggered at will by the consumer.
    // This is very similar to the crash below this one that causes UD2s (via replacing job pointers to UD2s), but it's not the same.
    for (auto ref = utility::scan(game, "E1 53 BD 4C 75 ?");
         ref.has_value();
         ref = utility::scan(*ref + 1, (game_end - (*ref + 1)) - 0x1000, "E1 53 BD 4C 75 ?"))
    {
        // Patch to 0x1337BEEF
        sus_constant_patches2.emplace_back(Patch::create(*ref, { 0xEF, 0xBE, 0x37, 0x13 }, true));
    }

    spdlog::info("[IntegrityCheckBypass]: Patched {} sus_constants! (RE9+)", sus_constant_patches2.size());

    // This is hidden within RenderTaskEnd. RenderTaskEnd is interleaved with legitimate game code and integrity checks. Entire function is obfuscated.
    // What they are doing is finding UD2 gadgets (even in the middle of instructions) around the game and replace random thread scheduler jobs
    // with pointers to the found UD2 function/gadget.
    // This pattern will likely change in the next update, we don't know what the invariants are yet without another sample.
    //auto thread_scheduler_corruptor = utility::scan(game, "48 89 74 08 08 48 89 F0");

    // Invariant that works through obfuscation. They don't obfuscate the epilogue of the block above the slow path conditional.
    // The xor rcx,rsp + call __security_check_cookie + vmovaps xmm6 sequence is compiler-generated and stable.
    // In new builds there was a sub rbp, rbp randomly inserted after the vmovaps, so we added a wildcard functionality to the signature scan to allow some instructions in between.
    
    const auto function_epilogue_sig = "48 31 E1 E8 ? ? ? ? *[5] C5 F8 28 B4 24 D0 01 00 00 *[5] 48 81 C4 E8 01 00 00";
    std::optional<uintptr_t> result{};
    size_t nop_size{};

    if (sdk::GameIdentity::get().is_re9()) {
    for (auto ref = utility::scan(game, function_epilogue_sig);
            ref.has_value();
            ref = utility::scan(*ref + 1, (game_end - (*ref + 1)) - 0x1000, function_epilogue_sig))
    {
        // We need to determine which one is the right one. First one is not the right one, it's not obfuscated.
        // Look for sequences of pops right after and skip.
        size_t pop_count = 0;
        utility::linear_decode((uint8_t*)*ref, 100, [&](utility::ExhaustionContext& ctx) -> bool {
            if (ctx.instrux.Category == ND_CAT_POP) {
                pop_count++;
            }

            // Stop at ret/int3/jmp
            if (ctx.instrux.Category == ND_CAT_RET || ctx.instrux.Category == ND_CAT_INTERRUPT || (ctx.instrux.BranchInfo.IsBranch && ctx.instrux.Category != ND_CAT_CALL)) {
                return false;
            }
            return true;
        });

        if (pop_count > 2) {
            spdlog::info("Skipping candidate at 0x{:X} due to high pop count: {}", *ref, pop_count);
            continue;
        }

        spdlog::info("Checking candidate at 0x{:X}, pop_count: {}", *ref, pop_count);

        // Linear decode past rets into the dispatch block. Look for the slow path discriminator:
        // a SETcc or CMOVcc instruction followed by a dispatch table access [reg + reg*8].
        // Old obfuscation used CMOVcc as the dispatch table load itself.
        // New obfuscation uses SETcc to set an index, then a separate MOV rXX, [rYY + rZZ*8].
        // The semantic proof that this is an obfuscated dispatch block is the presence of BOTH:
        //   1) a flag-dependent instruction (CMOVcc or SETcc)
        //   2) a SIB-indexed memory load with scale=8 (the 2-entry dispatch table)
        std::optional<uintptr_t> cond_addr{};
        size_t cond_nop_size{};
        bool has_dispatch_table = false;
        bool already_in_dispatch_block = false;

        utility::linear_decode((uint8_t*)*ref, 0x150, [&](utility::ExhaustionContext& ctx) -> bool {
            // CMOVcc pattern (old + variant): The cmov IS the dispatch selector.
            // Variant A: cmovcc rcx, [rip+disp32] (memory source, dispatch table load)
            // Variant B: cmovcc rcx, rbx (register source, both paths loaded via LEA)
            // In all observed variants the CMOVcc conditionally overwrites the default (clean)
            // path with the penalty path. NOPing it keeps the clean path unconditionally.
            // The structural context (epilogue match + pop_count <= 2 + ret-skip) is sufficient
            // to confirm this is an obfuscated dispatch, no operand-type check needed.
            if (ctx.instrux.Instruction == ND_INS_CMOVcc) {
                cond_addr = ctx.addr;
                cond_nop_size = ctx.instrux.Length;
                has_dispatch_table = true;
                return false;
            }

            // New pattern: SETcc sets an index register, then a separate MOV rXX, [rYY + rZZ*8]
            // reads from the 2-entry dispatch table using that index.
            if (ctx.instrux.Instruction == ND_INS_SETcc) {
                cond_addr = ctx.addr;
                cond_nop_size = ctx.instrux.Length;
            }

            // After finding a SETcc, look for the dispatch table access: MOV with [base + index*8]
            if (cond_addr.has_value() && ctx.instrux.Instruction == ND_INS_MOV) {
                for (uint8_t i = 0; i < ctx.instrux.OperandsCount; i++) {
                    const auto& op = ctx.instrux.Operands[i];
                    if (op.Type == ND_OP_MEM && op.Info.Memory.HasIndex && op.Info.Memory.HasBase && op.Info.Memory.Scale == 8) {
                        has_dispatch_table = true;
                        return false;
                    }
                }
            }

            // Skip past ret + garbage byte to continue into the next obfuscated block.
            if (ctx.instrux.Category == ND_CAT_RET) {
                if (already_in_dispatch_block) {
                    // We've already passed through one RET, the next RET would be the end of the dispatch block, so stop.
                    return false;
                }

                already_in_dispatch_block = true;
                ctx.addr += 1;
            }
            return true;
        });

        if (cond_addr.has_value() && has_dispatch_table) {
            result = cond_addr;
            nop_size = cond_nop_size;
            break;
        }
    }
    }

    // Fallback: UD2 writer anchor approach (works for MHSTORIES3 and other games where the
    // epilogue signature above doesn't match). The UD2 writer instruction 'mov [rax+rcx+8], rdx'
    // (48 89 ? 08 08) is unique or near-unique in the anti-tamper section. Searching backwards from it
    // for the SETcc + dispatch table load pattern finds the discriminator reliably.
#if 0
    if (!result) {
        spdlog::info("[IntegrityCheckBypass]: Epilogue scan failed, trying UD2 writer anchor approach...");

        for (auto ud2_ref = utility::scan(game, "48 89 ? 08 08");
             ud2_ref.has_value() && !result;
             ud2_ref = utility::scan(*ud2_ref + 1, (game_end - (*ud2_ref + 1)) - 0x1000, "48 89 ? 08 08"))
        {
            // Filter: the real UD2 writer uses SIB addressing: mov [base+index+disp8], reg.
            // ModR/M byte (offset +2) must have rm=100 (SIB follows) and mod=01 (8-bit disp).
            // False positives like mov [rdi+0x808],rax have rm=111 and mod=10 (32-bit disp).
            const uint8_t modrm = *reinterpret_cast<const uint8_t*>(*ud2_ref + 2);
            if ((modrm & 0xC7) != 0x44) { // mod=01, rm=100 -> SIB + disp8
                continue;
            }

            // Search backwards from the UD2 writer for the dispatch pattern:
            // [REX?] 0F 9x {ModR/M mod=11} [REX.W] 8B {ModR/M rm=100(SIB)} {SIB scale=8}
            // The SETcc sets an index (0 or 1), the MOV loads from a 2-entry dispatch table.
            const auto search_start = (*ud2_ref > 0x2000) ? (*ud2_ref - 0x2000) : (uintptr_t)game;
            const uint8_t* base = reinterpret_cast<const uint8_t*>(search_start);
            const size_t search_len = *ud2_ref - search_start;

            for (size_t i = 0; i < search_len; i++) {
                // Check for 0F 9x with the byte after having mod=11 (>= 0xC0)
                size_t setcc_off = 0;
                size_t setcc_len = 0;

                // Pattern A: no REX prefix on SETcc -> 0F 9? {mod=11}
                if (base[i] == 0x0F && (base[i+1] & 0xF0) == 0x90 && (base[i+2] & 0xC0) == 0xC0) {
                    setcc_off = i;
                    setcc_len = 3;
                }
                // Pattern B: REX prefix (40-4F) before SETcc -> 4? 0F 9? {mod=11}
                else if ((base[i] & 0xF0) == 0x40 && base[i+1] == 0x0F && (base[i+2] & 0xF0) == 0x90 && (base[i+3] & 0xC0) == 0xC0) {
                    setcc_off = i;
                    setcc_len = 4;
                }
                else {
                    continue;
                }

                // Now check if a MOV with SIB scale=8 follows within the next few bytes
                // (there may be 0-2 intervening bytes between the SETcc and the MOV).
                size_t dispatch_mov_end = 0;
                bool found_dispatch = false;
                for (size_t j = setcc_off + setcc_len; j < setcc_off + setcc_len + 4 && j + 3 < search_len; j++) {
                    // REX.W prefix (48-4F) followed by 8B (MOV), ModR/M with rm=100 (SIB), SIB with scale=8
                    if ((base[j] & 0xF0) == 0x40 && base[j+1] == 0x8B && (base[j+2] & 0x07) == 0x04 && (base[j+3] & 0xC0) == 0xC0) {
                        found_dispatch = true;
                        dispatch_mov_end = j + 4; // byte after the 4-byte MOV+SIB
                        break;
                    }
                }

                if (!found_dispatch) {
                    continue;
                }

                // Final verification: this must be a anti-tamper dispatch block, not normal game code.
                // anti-tamper dispatches always end with 'xchg [rsp], rXX; ret' (obfuscated indirect jmp).
                // Compilers never emit this pattern. Search forward from the dispatch MOV for:
                //   [REX?] 87 {ModR/M: mod=00, rm=100(SIB)} 24(SIB=[rsp]) C3(ret)
                bool has_xchg_ret = false;
                for (size_t k = dispatch_mov_end; k + 4 < search_len && k < dispatch_mov_end + 30; k++) {
                    size_t xo = k;
                    if ((base[xo] & 0xF0) == 0x40) xo++; // skip optional REX
                    if (xo + 3 < search_len &&
                        base[xo] == 0x87 && (base[xo+1] & 0xC7) == 0x04 && base[xo+2] == 0x24 && base[xo+3] == 0xC3) {
                        has_xchg_ret = true;
                        break;
                    }
                }

                if (has_xchg_ret) {
                    result = search_start + setcc_off;
                    nop_size = setcc_len;
                    spdlog::info("[IntegrityCheckBypass]: Found SETcc dispatch via UD2 writer anchor @ 0x{:X} ({}B), UD2 writer @ 0x{:X}",
                        *result, nop_size, *ud2_ref);
                    break;
                }
            }
        }
    }
#endif

    if (result) {
        spdlog::info("[IntegrityCheckBypass]: Found slow path discriminator @ 0x{:X} ({}B), patching...", *result, nop_size);
        // NOP the conditional. This forces the dispatch index to its default (clean) value:
        // - For SETcc: the target register keeps its restored value (0) from the surrounding obfuscation,
        //   so the dispatch table always selects index 0 (the clean path).
        // - For CMOVcc: the destination register keeps the value from the preceding MOV (the default path),
        //   preventing the conditional overwrite to the penalty path.
        std::vector<int16_t> nops{};
        nops.resize(nop_size, 0x90);
        static auto patch = Patch::create(*result, nops, true);
        spdlog::info("[IntegrityCheckBypass]: Patched slow path discriminator!");
    }
    
    // Hook this anyways as a backup plan.
    {
        if (!result) {
            spdlog::error("[IntegrityCheckBypass]: Could not find conditional move instruction for thread scheduler corruptor in RE9!");
            spdlog::error("[IntegrityCheckBypass]: Could not find thread scheduler corruptor in RE9!");
        }

        spdlog::warn("[IntegrityCheckBypass]: Attempting to hook JobQueue::SubmitDescriptor as a fallback for RE9. This may cause lag during integrity check jobs, but it should prevent crashes.");

        // Temporary workarounds for when none of that can be found
        // Temporarily needed on EGS and Japanese copies where obfuscation is different.
        // game will still lag but function with these.

        // This one is unnecessary and seems to be more unstable than mid-hooking the job callsites.
        /*auto ref = utility::scan(game, "41 B9 FF FF FF FF E8 ? ? ? ? 48 89 BE");
        auto fn = ref ? utility::calculate_absolute(*ref + 7) : std::optional<uintptr_t>{};

        if (fn) {
            g_submit_hook = safetyhook::create_inline(
                *fn,
                hk_JobQueue_SubmitDescriptor
            );

            spdlog::info("[IntegrityCheckBypass]: Hooked JobQueue::SubmitDescriptor in RE9 @ 0x{:X}!", *fn);
        }*/

        static std::vector<SafetyHookMid> callsites{};
        const auto candidate_pats = std::vector<std::string>{
            "? 8b ? 08 ? 8b ? 10 ? 8b ? 18 48 85 c9 0f 84 ? ? ? ? ff d0", // observed in RE9 PC, MHSTORIES 3
            "? 8b ? 08 ? 8b ? 10 ? 8b ? 18 48 85 c9 74 ? ff d0", // Rare path sometimes taken. seen in both.
        };

        for (const auto& pat : candidate_pats) {
            for (auto ref = utility::scan(utility::get_executable(), pat); 
                ref; 
                ref = utility::scan((*ref + 1), game_end - (*ref + 1), pat)) 
            {
                const auto dec = utility::decode_one((uint8_t*)(*ref));
                int reg = NDR_RDX; // default to rdx, which is the most common register used for the job descriptor pointer in observed patterns
                if (dec && dec->OperandsCount >= 2 && dec->Operands[1].Type == ND_OP_MEM) {
                    // determine register being used in right hand side (mem)
                    reg = dec->Operands[1].Info.Memory.Base;
                    spdlog::info("[IntegrityCheckBypass]: Found candidate call site for job submission with integrity check in RE9 @ 0x{:X}, using register {} for descriptor", *ref, reg);
                } else {
                    spdlog::warn("[IntegrityCheckBypass]: Found candidate call site for job submission with integrity check in RE9 @ 0x{:X}, but failed to decode register used for descriptor, defaulting to rdx", *ref);
                }

                switch (reg)
                {
                case NDR_RAX:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RAX>));
                    break;
                case NDR_RCX:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RCX>));
                    break;
                case NDR_RDX:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RDX>));
                    break;
                case NDR_RBX:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RBX>));
                    break;
                case NDR_RSP:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RSP>));
                    break;
                case NDR_RBP:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RBP>));
                    break;
                case NDR_RSI:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RSI>));
                    break;
                case NDR_RDI:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RDI>));
                    break;
                case NDR_R8:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R8>));
                    break;
                case NDR_R9:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R9>));
                    break;
                case NDR_R10:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R10>));
                    break;
                case NDR_R11:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R11>));
                    break;
                case NDR_R12:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R12>));
                    break;
                case NDR_R13:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R13>));
                    break;
                case NDR_R14:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R14>));
                    break;
                case NDR_R15:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_R15>));
                    break;
                default:
                    callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), &validate_job_func<NDR_RDX>));
                    break;
                };

                //callsites.emplace_back(safetyhook::create_mid((void*)(*ref + 4), validate_job_func
                spdlog::info("[IntegrityCheckBypass]: Hooked call site at 0x{:X}", *ref);
            }
        }
    }

    // Scan for PE header integrity check (thanks to SunBeam for pointing out this exists in RE9 and showing me where it is!)
    auto before_sig = "4C 89 ? 24 40 00 00 00 41 ?";
    bool patched_pe_header_check = false;

    for (auto ref = utility::scan(game, before_sig);
         ref.has_value();
         ref = utility::scan(*ref + 1, (game_end - (*ref + 1)) - 0x1000, before_sig))
    {
        spdlog::info("[IntegrityCheckBypass]: Checking candidate for PE header integrity check at 0x{:X}...", *ref);

        bool found_0x20 = false;
        bool found_0x28 = false;
        bool found = false;

        utility::linear_decode((uint8_t*)*ref, 0x200, [&](utility::ExhaustionContext& ctx) -> bool {
            const auto& ix = ctx.instrux;

            auto has_mem_operand_with_disp = [&](uint64_t disp) -> bool {
                for (uint8_t i = 0; i < ix.OperandsCount; i++) {
                    if (ix.Operands[i].Type == ND_OP_MEM &&
                        ix.Operands[i].Info.Memory.HasBase &&
                        ix.Operands[i].Info.Memory.HasDisp &&
                        ix.Operands[i].Info.Memory.Disp == disp)
                    {
                        return true;
                    }
                }
                return false;
            };

            if (!found_0x20 && has_mem_operand_with_disp(0x20)) {
                found_0x20 = true;
                return true;
            }

            if (found_0x20 && !found_0x28 && has_mem_operand_with_disp(0x28)) {
                found_0x28 = true;
                return true;
            }

            if (found_0x20 && found_0x28 && !found) {
                for (uint8_t i = 0; i < ix.OperandsCount; i++) {
                    if (ix.Operands[i].Type == ND_OP_MEM &&
                        ix.Operands[i].Info.Memory.HasBase &&
                        ix.Operands[i].Info.Memory.Base == NDR_RSP &&
                        ix.Operands[i].Info.Memory.HasDisp &&
                        ix.Operands[i].Info.Memory.Disp == 0x90)
                    {
                        found = true;
                        return false;
                    }
                }
            }

            // Stop at ret/int3/unconditional jmp (but not call)
            if (ix.Category == ND_CAT_RET || ix.Category == ND_CAT_INTERRUPT ||
                (ix.BranchInfo.IsBranch && !ix.BranchInfo.IsConditional && ix.Category != ND_CAT_CALL))
            {
                return false;
            }

            return true;
        });

        if (found) {
            spdlog::info("[IntegrityCheckBypass]: Found PE header integrity check at 0x{:X}!", *ref);

            static auto allocated_memory = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            memcpy(allocated_memory, (void*)GetModuleHandleA(nullptr), 0x1000);

            size_t pattern_byte_size = 0; // "4C 89 ? 24 40 00 00 00 41 ?"
            const auto patch_addr = *ref + 10;

            size_t reg = 0;
            bool found_register = false;

            // Emulate past this point and watch for when a register turns into the imagebase (0x140000000 in RE9).
            // This is the register we need to patch with our allocated memory.
            // It also lets us know how many bytes we actually need to NOP.
            // If we DON'T find it, we don't need to mindlessly patch this.
            // This only needs to be patched in the rare case someone actually modifies the PE header.
            utility::emulate(game, patch_addr, 15, [&](utility::ShemuContextExtended ctx) -> utility::ExhaustionResult {
                pattern_byte_size += ctx.ctx->ctx->Instruction.Length;

                // now check ALL THE REGISTERS.
                const auto regs = (uint64_t*)&ctx.ctx->ctx->Registers;
                for (size_t i = NDR_RAX; i <= NDR_R15; i++) {
                    if (regs[i] == (uintptr_t)game) {
                        spdlog::info("[IntegrityCheckBypass]: Found register containing image base: {}, at instruction 0x{:X}!", i, patch_addr + pattern_byte_size);
                        reg = i;
                        found_register = true;
                        return utility::ExhaustionResult::BREAK;
                    }
                }

                // Disallow memory writes so we don't break game state.
                if (ctx.next.writes_to_memory) {
                    return utility::ExhaustionResult::STEP_OVER; // yeet. swag. dab. no scope. big chungus.
                }

                // step over calls we don't care.
                if (ctx.next.ix.Category == ND_CAT_CALL) {
                    return utility::ExhaustionResult::STEP_OVER;
                }

                return utility::ExhaustionResult::CONTINUE;
            });

            // Decode the instruction at patch_addr to get the destination register
            /*const auto first_ix = utility::decode_one((uint8_t*)patch_addr);
            const auto reg = first_ix->Operands[0].Info.Register.Reg;
            const auto first_ix_len = first_ix->Length;

            // Decode the next instruction to know how many bytes to NOP
            const auto second_ix = utility::decode_one((uint8_t*)(patch_addr + first_ix_len));
            const auto second_ix_len = second_ix->Length;*/

            // Build movabs reg, allocated_memory using asmjit

            if (found_register) {
                using namespace asmjit;
                using namespace asmjit::x86;

                CodeHolder code{};
                code.init(Environment::host());
                Assembler a{&code};

                a.movabs(gpq(reg), (uintptr_t)allocated_memory);

                const auto& buf = code.textSection()->buffer();
                //const auto total_size = first_ix_len + second_ix_len;
                std::vector<uint8_t> raw(pattern_byte_size, 0x90);
                memcpy(raw.data(), buf.data(), buf.size());

                std::vector<int16_t> patch_bytes(raw.begin(), raw.end());
                static auto pe_header_patch = Patch::create(patch_addr, patch_bytes, true);


                spdlog::info("[IntegrityCheckBypass]: Patched PE header integrity check with movabs to 0x{:X} (reg: {})", (uintptr_t)allocated_memory, reg);
                patched_pe_header_check = true;
            } else {
                spdlog::error("[IntegrityCheckBypass]: Could not find register containing image base for PE header integrity check!");
            }

            break;
        }
    }

    if (!patched_pe_header_check) {
        spdlog::error("[IntegrityCheckBypass]: Could not find PE header integrity check!");
    }
}

void IntegrityCheckBypass::re9_heartbeat_bypass() {
    // let me explain what's happening here.
    // because the obfuscation has been randomized around the areas we've been patching so far (immediate_patch_re9, see commented out code)
    // I had become a bit fed up with manually fixing broken anti-tamper bypasses every update.
    // So I wrote an emulator that executed the RenderTaskEnd path (which contains anti-tamper code, especially the penalty code)
    // During my analysis of the trace, I found the conditional that decided between the penalty or the clean path.
    // So instead of patching that, I wanted to figure out WHAT caused that conditional to evaluate to "tampered" in the first place.
    // I found that, inside of a bunch of horrible obfuscated code, it was evaluating some value inside the renderer.
    // In this case it almost looked like the frame count.
    // I analyzed the memory region near this frame count and noticed 6 other values very close in value to the frame count, and they were all being updated
    // every 500ms or so to the actual frame count.
    // I noticed that when any of these frame counts were set to 0, the penalty path triggered and the game lagged to hell or crashed.
    // I then noticed that making these values equal to the frame count always made the clean path trigger, even if the integrity checks were triggered.
    // No patching necessary!
    if (sdk::GameIdentity::get().tdb_ver() < 82) {
        return;
    }
    static auto renderer_t = sdk::find_type_definition("via.render.Renderer");
    static auto get_RenderFrame = renderer_t != nullptr ? renderer_t->get_method("get_RenderFrame") : nullptr;
    auto renderer = sdk::get_native_singleton("via.render.Renderer");

    if (renderer != nullptr && renderer_t != nullptr && get_RenderFrame != nullptr) {
        static uint32_t* heartbeat_offset_start{nullptr};
        static std::vector<uintptr_t> candidates{};
        static uint32_t last_scan_frame = 0;
        static int confirmation_count = 0;
        static constexpr int CONFIRMATIONS_NEEDED = 3;
        static constexpr int32_t MAX_DISTANCE = 1000;
        static constexpr size_t HEARTBEAT_COUNT = 6;

        const auto frame_count = get_RenderFrame->call<uint32_t>(); // static func
        const auto renderer_addr = (uintptr_t)renderer;

        if (heartbeat_offset_start != nullptr) {
            // Confirmed, sync heartbeats to frame counter every frame
            for (size_t i = 0; i < HEARTBEAT_COUNT; i++) {
                heartbeat_offset_start[i] = frame_count;
            }
        } else if (frame_count > 100 && frame_count != last_scan_frame) {
            // Debug for RE9 (known to be at 0x3328)
#if 0
            for (size_t i = 0; i < HEARTBEAT_COUNT; i++) {
                auto val = *(uint32_t*)(renderer_addr + 0x3328 + i * 4);
                spdlog::info("[IntegrityCheckBypass] Heartbeat candidate {}: {} (diff: {}), actual: {}", i, val, frame_count - val, frame_count);
            }
#endif

            last_scan_frame = frame_count;

            // Two detection modes for the heartbeat cluster:
            // Normal: sentinel(1), 6 valid heartbeats, sentinel(0)
            // Early:  sentinel(1), 0 (heartbeat not yet written), 5 valid, sentinel(0)
            // In RE9, heartbeat[0] doesn't get its first write until ~frame 930.
            // The anti-tamper starts corrupting job pointers well before that.
            std::vector<uintptr_t> this_frame{};
            for (size_t i = 0x2000; i + HEARTBEAT_COUNT * 4 <= 0x4000; i += sizeof(uint32_t)) {
                try {
                    auto* ints = reinterpret_cast<uint32_t*>(renderer_addr + i);
                    if (ints[-1] != 1) {
                        continue; // sentinel before cluster must be 1
                    }
                    if (ints[HEARTBEAT_COUNT] != 0) {
                        continue; // sentinel after cluster must be 0
                    }

                    // Check if all 6 are valid heartbeats
                    bool all_valid = true;
                    // Check if ints[0] == 0 (unwritten) and ints[1..5] are valid
                    bool early_detect = (ints[0] == 0);

                    for (size_t j = 0; j < HEARTBEAT_COUNT; j++) {
                        auto val = ints[j];
                        bool in_range = val > 0 && val <= frame_count && (frame_count - val) < (uint32_t)MAX_DISTANCE;
                        if (!in_range) {
                            all_valid = false;
                        }
                        // For early detect: ints[1..5] must all be in range
                        if (j > 0 && !in_range) {
                            early_detect = false;
                        }
                    }

                    if (all_valid || early_detect) {
                        this_frame.push_back(renderer_addr + i);
                    }
                } catch (...) {}
            }

            if (candidates.empty()) {
                // First scan, seed candidates
                candidates = std::move(this_frame);
                confirmation_count = 1;
            } else {
                // Intersect with previous candidates, only keep offsets
                // that match across multiple frames
                std::vector<uintptr_t> intersection{};
                for (auto addr : candidates) {
                    if (std::find(this_frame.begin(), this_frame.end(), addr) != this_frame.end()) {
                        intersection.push_back(addr);
                    }
                }
                candidates = std::move(intersection);
                confirmation_count++;

                if (candidates.size() == 1 && confirmation_count >= CONFIRMATIONS_NEEDED) {
                    heartbeat_offset_start = (uint32_t*)candidates[0];
                    spdlog::info("[IntegrityCheckBypass] Found heartbeat cluster at renderer+0x{:X} after {} confirmations at frame count {}, syncing it to frame count every frame now",
                        (uintptr_t)heartbeat_offset_start - renderer_addr, confirmation_count, frame_count);
                } else if (candidates.empty()) {
                    // Lost all candidates, restart
                    confirmation_count = 0;
                    spdlog::warn("[IntegrityCheckBypass] Heartbeat candidates lost, restarting scan");
                }
            }
        }
    }

}

// The new-style stack destroyer is a multi-instruction obfuscated gadget. The pattern scan above
// points at the middle of that gadget, and patching there with RET would pop whatever the gadget
// already pushed (not a return address). Resolve the gadget's entry point instead.
//
// These builds have no PE unwind info (no .pdata), so find_function_start/get_disassembly_behind
// cannot be used. The obfuscator chains gadgets with RET (it pushes the next gadget address), so
// the entry is the byte after the nearest preceding RET that decodes forward into the candidate.
static std::optional<uintptr_t> find_stack_destroyer_entry(uintptr_t candidate, uintptr_t module_base) {
    constexpr uintptr_t max_back = 0x200;

    const auto limit = candidate > max_back ? candidate - max_back : module_base;

    for (auto addr = candidate; addr > limit; --addr) {
        if (*(uint8_t*)(addr - 1) != 0xC3) {
            continue;
        }

        bool reached_candidate = false;

        utility::linear_decode((uint8_t*)addr, (candidate - addr) + 0x20, [&](utility::ExhaustionContext& ctx) -> bool {
            if (ctx.addr == candidate) {
                reached_candidate = true;
                return false;
            }

            return ctx.addr < candidate;
        });

        if (reached_candidate) {
            return addr;
        }
    }

    return std::nullopt;
}

// 1.42.0.2 carries a *second* stack destroyer that the .udata pivot above never matches. It is a
// tail-call trampoline in .data:
//
//   mov     rax, rcx            ; 48 89 C8
//   mov     qword ptr [rsp], 0  ; 48 C7 04 24 00 00 00 00   <- zeroes the return slot
//   mov     ecx, 0              ; variant A ... or: mov rcx, rdx / mov edx, 0 ... variant B
//   jmp     rax                 ; FF E0
//
// RCX is the real callee. This is the gadget on the faulting stack -- the ud2 at 0xAE46DDF sits
// immediately after its `jmp rax` -- and the crash context matches variant B exactly: RAX = old RCX
// (0), RCX = old RDX (0x3F32D2192AE10000), RDX = 0. So the caller computed a NULL callee and
// `jmp rax` faulted at RIP=0.
//
// Diagnostic only: patching the entry with RET would skip the callee entirely, so mid-hook it and
// log the callee + caller instead.
static constexpr auto STACK_DESTROYER_TRAMPOLINE_HEAD = "48 89 C8 48 C7 04 24 00 00 00 00";
static constexpr size_t STACK_DESTROYER_TRAMPOLINE_HEAD_SIZE = 11;

static constexpr uint8_t STACK_DESTROYER_TRAMPOLINE_TAIL_A[] = {0xB9, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xE0};
static constexpr uint8_t STACK_DESTROYER_TRAMPOLINE_TAIL_B[] = {0x48, 0x89, 0xD1, 0xBA, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xE0};

enum class StackDestroyerTrampolineKind {
    MovEcxZeroJmpRax, // "mov ecx, 0; jmp rax"
    MovRcxRdxJmpRax,  // "mov rcx, rdx; mov edx, 0; jmp rax"
    Unknown,
};

struct StackDestroyerTrampoline {
    uintptr_t address{};
    StackDestroyerTrampolineKind kind{StackDestroyerTrampolineKind::Unknown};
};

static std::string_view stack_destroyer_trampoline_kind_name(StackDestroyerTrampolineKind kind) {
    switch (kind) {
    case StackDestroyerTrampolineKind::MovEcxZeroJmpRax:
        return "mov ecx,0; jmp rax";
    case StackDestroyerTrampolineKind::MovRcxRdxJmpRax:
        return "mov rcx,rdx; mov edx,0; jmp rax";
    default:
        return "unrecognised tail";
    }
}

// utility::scan() only returns the first hit, so walk the module one byte at a time. The 11-byte
// head alone is not enough (it also occurs inside longer instruction streams), so classify by the
// tail that follows it.
static std::vector<StackDestroyerTrampoline> find_stack_destroyer_trampolines(HMODULE game) {
    std::vector<StackDestroyerTrampoline> out{};

    const auto size = utility::get_module_size(game);

    if (!size) {
        return out;
    }

    const auto end = (uintptr_t)game + *size;
    auto start = (uintptr_t)game;

    while (start < end) {
        const auto head = utility::scan(start, end - start, STACK_DESTROYER_TRAMPOLINE_HEAD);

        if (!head) {
            break;
        }

        start = *head + 1;

        const auto tail = (const uint8_t*)(*head + STACK_DESTROYER_TRAMPOLINE_HEAD_SIZE);

        if (IsBadReadPtr(tail, sizeof(STACK_DESTROYER_TRAMPOLINE_TAIL_B))) {
            continue;
        }

        if (memcmp(tail, STACK_DESTROYER_TRAMPOLINE_TAIL_A, sizeof(STACK_DESTROYER_TRAMPOLINE_TAIL_A)) == 0) {
            out.push_back({*head, StackDestroyerTrampolineKind::MovEcxZeroJmpRax});
        } else if (memcmp(tail, STACK_DESTROYER_TRAMPOLINE_TAIL_B, sizeof(STACK_DESTROYER_TRAMPOLINE_TAIL_B)) == 0) {
            out.push_back({*head, StackDestroyerTrampolineKind::MovRcxRdxJmpRax});
        } else {
            out.push_back({*head, StackDestroyerTrampolineKind::Unknown});
        }
    }

    return out;
}

static uintptr_t s_stack_destroyer_trampoline_a{0};
static uintptr_t s_stack_destroyer_trampoline_b{0};

static void log_stack_destroyer_trampoline(uintptr_t address, safetyhook::Context& ctx) {
    static std::atomic<uint32_t> s_hits{0};

    const auto hit = s_hits.fetch_add(1, std::memory_order_relaxed);
    const auto callee = (uintptr_t)ctx.rcx;
    const auto rsp = (uintptr_t)ctx.rsp;

    // This sits on the game's hot path, so bound the volume. A NULL callee is the fatal `jmp rax`
    // and is always logged.
    if (hit >= 32 && callee != 0) {
        return;
    }

    spdlog::error("[IntegrityCheckBypass]: stack-destroyer trampoline {} hit #{}: rcx(callee) {} rdx 0x{:X} rax 0x{:X} rsp 0x{:X}",
        describe_address(address), hit, describe_address(callee), (uintptr_t)ctx.rdx, (uintptr_t)ctx.rax, rsp);

    // The callers are on the stack the trampoline is about to destroy, so capture it now.
    const auto deep = hit < 4 || callee == 0;

    if (deep) {
        log_stack_scan(rsp, 0x40, "trampoline");
    }

    if (deep) {
        utility::exceptions::dump_callstack(nullptr);
    }
}

static void stack_destroyer_trampoline_a_hook(safetyhook::Context& ctx) {
    log_stack_destroyer_trampoline(s_stack_destroyer_trampoline_a, ctx);
}

static void stack_destroyer_trampoline_b_hook(safetyhook::Context& ctx) {
    log_stack_destroyer_trampoline(s_stack_destroyer_trampoline_b, ctx);
}

static std::vector<safetyhook::MidHook> s_stack_destroyer_trampoline_hooks{};

static void hook_stack_destroyer_trampolines(HMODULE game) {
    if (!s_stack_destroyer_trampoline_hooks.empty()) {
        return;
    }

    const auto trampolines = find_stack_destroyer_trampolines(game);

    if (trampolines.empty()) {
        spdlog::error("[IntegrityCheckBypass]: No 1.42.0.2 stack-destroyer trampolines found!");
        return;
    }

    for (const auto& trampoline : trampolines) {
        spdlog::info("[IntegrityCheckBypass]: stack-destroyer trampoline at {} ({})", describe_address(trampoline.address),
            stack_destroyer_trampoline_kind_name(trampoline.kind));

        // The image is packed, so the bytes in memory are what actually executes.
        if (const auto original = utility::get_original_bytes(trampoline.address); original && !original->empty()) {
            std::string bytes{};

            for (const auto b : *original) {
                bytes += fmt::format("{:02X} ", b);
            }

            spdlog::warn("[IntegrityCheckBypass]:     on-disk bytes differ from runtime: {}", bytes);
        }
    }

    for (const auto& trampoline : trampolines) {
        const auto is_variant_a = trampoline.kind == StackDestroyerTrampolineKind::MovEcxZeroJmpRax;
        const auto is_variant_b = trampoline.kind == StackDestroyerTrampolineKind::MovRcxRdxJmpRax;

        if (!is_variant_a && !is_variant_b) {
            continue;
        }

        if (is_variant_a) {
            s_stack_destroyer_trampoline_a = trampoline.address;
        } else {
            s_stack_destroyer_trampoline_b = trampoline.address;
        }

        auto hook = safetyhook::create_mid((void*)trampoline.address,
            is_variant_a ? &stack_destroyer_trampoline_a_hook : &stack_destroyer_trampoline_b_hook);

        if (!hook) {
            spdlog::error("[IntegrityCheckBypass]: Failed to hook stack-destroyer trampoline at 0x{:X}!", trampoline.address);
            continue;
        }

        s_stack_destroyer_trampoline_hooks.emplace_back(std::move(hook));

        // A mid-hook rewrites the entry with a jump. Read it back so "no hit" cannot be confused
        // with "the patch was reverted before the gadget ran".
        std::string live_bytes{};

        for (size_t i = 0; i < 8; ++i) {
            live_bytes += fmt::format("{:02X} ", ((const uint8_t*)trampoline.address)[i]);
        }

        spdlog::info("[IntegrityCheckBypass]:     live bytes at 0x{:X}: {}", trampoline.address, live_bytes);
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked {} stack-destroyer trampoline(s) for diagnostics.",
        s_stack_destroyer_trampoline_hooks.size());
}

// The ud2 decoy at 0x14AE46DDF is materialised by exactly five game functions, all the same
// integrity scanner. Each walks a table of 0x400-byte entries and, for every entry whose first
// dword is 0x4cbd53e1 and whose first 16 bytes XOR to zero against a fixed key, calls a per-entry
// handler at [rdi+0x650] passing the ud2 address in rcx:
//
//   vmovdqa xmm6, [key]
//   lea     rsi, [rip+...]        # 0x14AE46DDF
// .loop:
//   cmp     dword [rdi], 0x4cbd53e1
//   jne     .next
//   vpxor   xmm0, xmm6, [rdi]
//   vptest  xmm0, xmm0
//   jne     .next
//   mov     rcx, rsi              # rcx = the ud2 address
//   call    qword ptr [rdi+0x650]  <- NULL/ud2 target here is the obvious crash candidate
// .next:
//   add     rdi, 0x400
//   cmp     rdi, rbx
//   jb      .loop
//
// Diagnostic only: log the handler target (and the entry) so a NULL or ud2 handler is visible.
// Anchor on the wildcard-free magic compare (`cmp dword [rdi], 0x4cbd53e1`); the `lea rsi,[ud2]`
// sits 7 bytes before it and the per-entry handler call ~22 bytes after. Wildcards do NOT survive
// utility::scan(start, length, pattern): the wildcard form matched 0 times at runtime while the same
// 5 sites match in the file, and the wildcard-free trampoline scan worked.
static constexpr auto STACK_DESTROYER_SCANNER = "81 3F E1 53 BD 4C";
static constexpr uintptr_t STACK_DESTROYER_SCANNER_LEA_BACK = 7;
static constexpr auto STACK_DESTROYER_SCANNER_CALL = "FF 97 50 06 00 00";

static void log_stack_destroyer_scanner_call(safetyhook::Context& ctx) {
    static std::atomic<uint32_t> s_hits{0};

    const auto entry = (uintptr_t)ctx.rdi;
    const auto target = IsBadReadPtr((void*)(entry + 0x650), sizeof(uintptr_t)) ? 0 : *(const uintptr_t*)(entry + 0x650);
    const auto hit = s_hits.fetch_add(1, std::memory_order_relaxed);

    // A NULL or non-module handler is the crash candidate; the normal handler is just noise.
    const auto suspicious = target == 0 || !utility::get_module_within(target).has_value();

    if (hit >= 32 && !suspicious) {
        return;
    }

    spdlog::error("[IntegrityCheckBypass]: stack-destroyer scanner call #{}: entry 0x{:X} rcx(ud2) 0x{:X} handler {}", hit, entry,
        (uintptr_t)ctx.rcx, describe_address(target));

    if (suspicious) {
        log_stack_scan((uintptr_t)ctx.rsp, 0x40, "scanner");
    }
}

static std::vector<safetyhook::MidHook> s_stack_destroyer_scanner_hooks{};

static void hook_stack_destroyer_scanner_calls(HMODULE game) {
    if (!s_stack_destroyer_scanner_hooks.empty()) {
        return;
    }

    const auto size = utility::get_module_size(game);

    if (!size) {
        return;
    }

    const auto end = (uintptr_t)game + *size;
    auto start = (uintptr_t)game;

    while (start < end) {
        const auto anchor = utility::scan(start, end - start, STACK_DESTROYER_SCANNER);

        if (!anchor) {
            break;
        }

        start = *anchor + 1;

        const auto scanner = *anchor - STACK_DESTROYER_SCANNER_LEA_BACK;

        // Must be `lea rsi, [rip+...]` and the wildcard displacement must land on a ud2.
        if (IsBadReadPtr((void*)scanner, 7) || *(const uint8_t*)(scanner + 1) != 0x8D || *(const uint8_t*)(scanner + 2) != 0x35) {
            continue;
        }

        const auto ud2_target = scanner + 7 + *(const int32_t*)(scanner + 3);

        if (IsBadReadPtr((void*)ud2_target, 2) || *(const uint16_t*)ud2_target != 0x0B0F) {
            continue;
        }

        // The handler call sits a few instructions further down the same loop body.
        const auto call = utility::scan(*anchor, 0x40, STACK_DESTROYER_SCANNER_CALL);

        if (!call) {
            spdlog::error("[IntegrityCheckBypass]: stack-destroyer scanner at {} has no handler call!", describe_address(scanner));
            continue;
        }

        spdlog::info("[IntegrityCheckBypass]: stack-destroyer scanner at {} (ud2 0x{:X}) calls handler at {}",
            describe_address(scanner), ud2_target, describe_address(*call));

        auto hook = safetyhook::create_mid((void*)*call, &log_stack_destroyer_scanner_call);

        if (!hook) {
            spdlog::error("[IntegrityCheckBypass]: Failed to hook stack-destroyer scanner handler call at 0x{:X}!", *call);
            continue;
        }

        s_stack_destroyer_scanner_hooks.emplace_back(std::move(hook));
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked {} stack-destroyer scanner handler call(s).",
        s_stack_destroyer_scanner_hooks.size());
}

// Debug registers are per-thread, so a thread created after arming is uncovered. Re-arm only the
// newcomers for a while from a throwaway thread; arm_stack_write_watchpoints() ignores TIDs it has
// already seen, so this stays cheap.
static DWORD WINAPI stack_watch_rearm_thread(LPVOID) {
    const auto deadline = GetTickCount64() + 20000;

    while (GetTickCount64() < deadline && !s_stack_watch_disarmed.load(std::memory_order_relaxed)) {
        arm_stack_write_watchpoints(false);
        Sleep(20);
    }

    return 0;
}

static std::vector<std::unique_ptr<Patch>> s_early_trampoline_patches{};

// The 1.42.0.2 tail-call trampolines wipe the return slot and then `jmp rax` into the ud2 at
// 0x14AE46DDF, which the VM catches as an exception (the faulting stack carries an EXCEPTION_RECORD
// with ExceptionCode 0xC000001D and ExceptionAddress 0x14AE46DDF). Only the destructive
// `mov qword [rsp],0` is removed -- replacing the entry with RET would skip the trap the VM expects.
static void neutralize_stack_destroyer_trampolines(HMODULE game) {
    static constexpr uint8_t mov_rsp_zero[] = {0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00};

    // trampoline+3 in both the `mov ecx,0` and `mov rcx,rdx; mov edx,0` variants, byte-verified for
    // 1.42.0.2.
    for (const auto rva : {0xAE46DBB, 0xAE46DCD}) {
        const auto address = (uintptr_t)game + rva;

        if (IsBadReadPtr((void*)address, sizeof(mov_rsp_zero)) || memcmp((const void*)address, mov_rsp_zero, sizeof(mov_rsp_zero)) != 0) {
            diag_log(fmt::format("trampoline neutralise: 0x{:X} is not `mov qword [rsp],0`, skipping", address));
            continue;
        }

        s_early_trampoline_patches.emplace_back(Patch::create(address, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90}, true));

        // If the trampoline still turns out to be the cause we need to know whether the patch stuck.
        std::string live{};

        for (size_t i = 0; i < 8; ++i) {
            live += fmt::format("{:02X} ", ((const uint8_t*)address)[i]);
        }

        diag_log(fmt::format("trampoline neutralise: 0x{:X} now holds {}", address, live));
    }
}

// Everything here has to run before the game touches the trampolines and before the frame at
// 0x1850C3D8 is built: `window at arm time` proved that frame is already fully populated by the time
// remove_stack_destroyer() runs, so anything installed there only observes the aftermath.
void IntegrityCheckBypass::early_mhwilds_diagnostics() {
    // This runs on startup_thread before the REFramework constructor, which is where the log file is
    // created. A fault in here means no log at all, so nothing may escape and the handler that makes
    // the DR self-test safe has to exist first.
    diag_log("early phase: entered");
    early_log_open();
    diag_log("early phase: raw early log opened");

    try {
        if (!sdk::GameIdentity::get().is_mhwilds()) {
            diag_log("early phase: not mhwilds, skipping");
            s_early_phase = false;
            return;
        }

        const auto game = utility::get_executable();

        if (game == nullptr) {
            diag_log("early phase: no executable module");
            s_early_phase = false;
            return;
        }

        // Must come first: run_dr_selftest() raises a data-breakpoint trap, and an unhandled
        // single-step terminates the process before the log file exists.
        diag_log("early phase: installing first-chance vectored handler");
        init_first_chance_exception_logger();

        diag_log("early phase: neutralising trampolines");
        neutralize_stack_destroyer_trampolines(game);

        diag_log("early phase: arming stack write watchpoints");
        arm_stack_write_watchpoints();

        diag_log("early phase: starting re-arm thread");
        CreateThread(nullptr, 0, stack_watch_rearm_thread, nullptr, 0, nullptr);

        diag_log("early phase: done");
    } catch (...) {
        diag_log("early phase: caught a C++ exception; continuing without early diagnostics");
        s_early_phase = false;
    }
}

void IntegrityCheckBypass::remove_stack_destroyer() {
    flush_early_log();

    spdlog::info("[IntegrityCheckBypass]: Searching for stack destroyer...");

    const auto game = utility::get_executable();

    if (game == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: "
                      "Could not obtain executable module!");
        return;
    }
    const auto module_base = reinterpret_cast<uintptr_t>(game);

    // The .data trampoline below is the one that actually faults, so hook it independently of
    // whether the .udata pivot is found (or matched) at all.
    if (sdk::GameIdentity::get().is_mhwilds()) {
        hook_stack_destroyer_trampolines(game);
        hook_stack_destroyer_scanner_calls(game);

        // Catch the store that wipes the return slot, and neutralise it in place.
        arm_stack_write_watchpoints();
    }

    const auto legacy = utility::scan(game, "48 89 11 "
                                            "48 C7 04 24 00 00 00 00 "
                                            "48 81 C4 28 01 00 00");

    if (legacy) {
        spdlog::info("[IntegrityCheckBypass]: "
                     "Found legacy stack destroyer at RVA 0x{:X}",
            *legacy - module_base);

        static auto patch = Patch::create(*legacy, {0xC3}, true);

        spdlog::info("[IntegrityCheckBypass]: "
                     "Patched legacy stack destroyer!");
        return;
    }

    // Newer games (MHWilds) no longer contain the SF6 shape above. The stack destroyer moved into
    // the obfuscated, self-modifying .udata section and is now expressed as a stack-pivot gadget
    // that rewrites the return address on the stack:
    //
    // mov     rbx, [rsp]
    // mov     qword ptr [rsp], 0
    // sub     qword ptr [rsp], rbp
    // mov     rbp, [rsp]
    // lea     rdx, [rip+...]
    // lea     rsp, [rsp+8]
    // xchg    qword ptr [rsp], rdx
    // add     rsp, -8
    //
    // This pattern only points at the middle of the gadget. Patching there would RET into whatever
    // the gadget has already pushed, so resolve the real entry first.
    const auto candidate = utility::scan(game, "48 8B 1C 24 "
                                               "48 C7 04 24 00 00 00 00 "
                                               "48 29 2C 24 "
                                               "48 8B 2C 24 "
                                               "48 8D 15 ? ? ? ? "
                                               "48 8D A4 24 08 00 00 00 "
                                               "48 87 14 24 "
                                               "48 83 C4 F8");

    if (!candidate) {
        spdlog::error("[IntegrityCheckBypass]: "
                      "Could not find legacy or 1.42.0.2 "
                      "stack-destroyer candidate!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: "
                 "Found 1.42.0.2 stack-destroyer candidate "
                 "at RVA 0x{:X}",
        *candidate - module_base);

    if (const auto entry = find_stack_destroyer_entry(*candidate, module_base)) {
        spdlog::info("[IntegrityCheckBypass]: "
                     "Resolved 1.42.0.2 stack destroyer entry "
                     "at RVA 0x{:X}",
            *entry - module_base);
    } else {
        spdlog::warn("[IntegrityCheckBypass]: "
                     "Could not resolve 1.42.0.2 stack destroyer entry");
    }

    // Diagnostic-only. The on-disk image does not match what the packer executes at runtime: .udata
    // is MEM_WRITE and gets rewritten, so the entry/gadget we resolve statically is not necessarily
    // the code that runs. Patching it corrupts packer/VM setup (crash within ~300ms of the patch).
    // Dump the live bytes around the candidate so the real target can be identified from the running
    // process instead of from the file.
    const auto dump_start = *candidate - 0x10;
    const auto dump_end = *candidate + 0x30;

    std::stringstream bytes{};
    bytes << std::hex << std::setfill('0');

    for (auto addr = dump_start; addr < dump_end; ++addr) {
        bytes << std::setw(2) << (uint32_t)*(uint8_t*)addr << " ";
    }

    spdlog::info("[IntegrityCheckBypass]: "
                 "Runtime bytes RVA 0x{:X} - 0x{:X}: {}",
        dump_start - module_base, dump_end - module_base, bytes.str());

    spdlog::warn("[IntegrityCheckBypass]: "
                 "1.42.0.2 stack-destroyer candidate located; "
                 "diagnostic-only mode, no patch applied!");
}

void IntegrityCheckBypass::setup_pristine_syscall() {
    if (s_pristine_protect_virtual_memory != nullptr) {
        spdlog::info("[IntegrityCheckBypass]: NtProtectVirtualMemory already setup!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Copying pristine NtProtectVirtualMemory...");

    const auto ntdll_base = GetModuleHandleA("ntdll.dll");

    if (ntdll_base == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find ntdll!");
        return;
    }

    auto nt_protect_virtual_memory = (NtProtectVirtualMemory_t)GetProcAddress(ntdll_base, "NtProtectVirtualMemory");
    if (nt_protect_virtual_memory == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find NtProtectVirtualMemory!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Found NtProtectVirtualMemory at 0x{:X}", (uintptr_t)nt_protect_virtual_memory);

    if (*(uint8_t*)nt_protect_virtual_memory == 0xE9) {
        spdlog::info("[IntegrityCheckBypass]: Found jmp at 0x{:X}, resolving...", (uintptr_t)nt_protect_virtual_memory);
        nt_protect_virtual_memory = (decltype(nt_protect_virtual_memory))utility::calculate_absolute((uintptr_t)nt_protect_virtual_memory + 1);
    }

    s_og_protect_virtual_memory = nt_protect_virtual_memory;

    // Mark the original VirtualProtect READ_WRITE_EXECUTE so if anything tries to restore the old protection, it will revert to this
    // incase trying to modify the protection after it is hooked causes a crash
    DWORD old_nt_protect_virtual_memory_protect{};
    VirtualProtect(nt_protect_virtual_memory, 256, PAGE_EXECUTE_READWRITE, &old_nt_protect_virtual_memory_protect);

    s_pristine_protect_virtual_memory = (decltype(s_pristine_protect_virtual_memory))VirtualAlloc(nullptr, 256, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    try {
        memcpy(s_pristine_protect_virtual_memory, nt_protect_virtual_memory, 256);
    } catch(...) {
        spdlog::error("[IntegrityCheckBypass]: Could not copy new instructions to pristine NtProtectVirtualMemory!");
    }

    spdlog::info("[IntegrityCheckBypass]: Copied NtProtectVirtualMemory to 0x{:X}", (uintptr_t)s_pristine_protect_virtual_memory);
}

// hahahah i hate this
void IntegrityCheckBypass::fix_virtual_protect() try {
    spdlog::info("[IntegrityCheckBypass]: Fixing VirtualProtect...");

    setup_pristine_syscall(); // Called earlier in DllMain

    // Hook VirtualProtect
    s_virtual_protect_hook = std::make_unique<FunctionHookMinHook>(VirtualProtect, (uintptr_t)virtual_protect_hook);
    if (!s_virtual_protect_hook->create()) {
        spdlog::error("[IntegrityCheckBypass]: Could not hook VirtualProtect!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked VirtualProtect!");
} catch(...) {
    spdlog::error("[IntegrityCheckBypass]: Could not fix VirtualProtect!");
}

BOOL WINAPI IntegrityCheckBypass::virtual_protect_impl(LPVOID lpAddress, SIZE_T dwSize, DWORD flNewProtect, PDWORD lpflOldProtect) {
    static const auto this_process = GetCurrentProcess();

    LPVOID address_to_protect = lpAddress;
    NTSTATUS result = s_og_protect_virtual_memory(this_process, (PVOID*)&address_to_protect, &dwSize, flNewProtect, lpflOldProtect);

    constexpr NTSTATUS STATUS_INVALID_PAGE_PROTECTION = 0xC0000045;

    // recreated from kernelbase to be correct
    if (result == STATUS_INVALID_PAGE_PROTECTION) {
        using RtlFlushSecureMemoryCache_t = BOOLEAN (NTAPI*)(PVOID, SIZE_T);
        static const auto rtl_flush_secure_memory_cache = (RtlFlushSecureMemoryCache_t)GetProcAddress(GetModuleHandleA("ntdll.dll"), "RtlFlushSecureMemoryCache");

        if (rtl_flush_secure_memory_cache != nullptr) {
            if (NT_SUCCESS(rtl_flush_secure_memory_cache(address_to_protect, dwSize))) {
                result = s_og_protect_virtual_memory(this_process, (PVOID*)&address_to_protect, &dwSize, flNewProtect, lpflOldProtect);

                if ((result & 0x80000000) == 0) {
                    return TRUE;
                }
            }
        }
    }

    if (!NT_SUCCESS(result)) {
        spdlog::error("[IntegrityCheckBypass]: NtProtectVirtualMemory(-1, {:x}, {:x}, {:x}, {:x}) failed with {:x}", (uintptr_t)address_to_protect, dwSize, flNewProtect, (uintptr_t)lpflOldProtect, (uint32_t)result);
    }
    
    return NT_SUCCESS(result);
}

// This allows our calls to VirtualProtect to go through without being hindered by... something.
BOOL WINAPI IntegrityCheckBypass::virtual_protect_hook(LPVOID lpAddress, SIZE_T dwSize, DWORD flNewProtect, PDWORD lpflOldProtect) try {
    static bool once = true;
    if (once) {
        spdlog::info("[IntegrityCheckBypass]: VirtualProtect called");
        once = false;
    }

    try {
        if (memcmp(s_og_protect_virtual_memory, s_pristine_protect_virtual_memory, 32) != 0) {
            spdlog::warn("[IntegrityCheckBypass]: Original NtProtectVirtualMemory has been modified! Attempting to restore...");

            bool needs_protection_fix = false;

            try {
                memcpy(s_og_protect_virtual_memory, s_pristine_protect_virtual_memory, 32);

                if (memcmp(s_og_protect_virtual_memory, s_pristine_protect_virtual_memory, 32) == 0) {
                    spdlog::info("[IntegrityCheckBypass]: Successfully restored NtProtectVirtualMemory");
                } else {
                    needs_protection_fix = true;
                    spdlog::error("[IntegrityCheckBypass]: Could not restore NtProtectVirtualMemory without changing protection!");
                }
            } catch(...) {
                needs_protection_fix = true;
                spdlog::error("[IntegrityCheckBypass]: Could not restore NtProtectVirtualMemory without changing protection! Attempting to restore protection anyway...");
            }

            if (needs_protection_fix) try {
                spdlog::info("[IntegrityCheckBypass]: Attempting to restore NtProtectVirtualMemory protection");

                DWORD old{};

                // Now this is a huge assumption that the hook that was placed on NtProtectVirtualMemory
                // does not prevent calling it on itself, which is a very dangerous assumption to make.
                // Usually it fails if it's attempted to be called on the executable's memory, but not other modules.
                // However I have tested it and it *does* work, so we will roll with that for now, until it doesn't.
                if (virtual_protect_impl((void*)((uintptr_t)s_og_protect_virtual_memory - 1), 33, PAGE_EXECUTE_READWRITE, &old)) {
                    memcpy(s_og_protect_virtual_memory, s_pristine_protect_virtual_memory, 32);
                    virtual_protect_impl((void*)((uintptr_t)s_og_protect_virtual_memory - 1), 33, old, &old);

                    spdlog::info("[IntegrityCheckBypass]: Restored NtProtectVirtualMemory");
                }
            } catch(...) {
                spdlog::error("[IntegrityCheckBypass]: Could not restore NtProtectVirtualMemory protection!");
            }
        }
    } catch(...) {
        spdlog::error("[IntegrityCheckBypass]: Failed to verify NtProtectVirtualMemory integrity!");
    }

    return virtual_protect_impl(lpAddress, dwSize, flNewProtect, lpflOldProtect);
} catch(...) {
    spdlog::error("[IntegrityCheckBypass]: VirtualProtect hook failed! falling back to original");
    return s_virtual_protect_hook->get_original<decltype(virtual_protect_hook)>()(lpAddress, dwSize, flNewProtect, lpflOldProtect);
}

void IntegrityCheckBypass::hook_add_vectored_exception_handler() {
    if (sdk::GameIdentity::get().tdb_ver() < 73) return;
    spdlog::info("[IntegrityCheckBypass]: Hooking AddVectoredExceptionHandler...");

    s_add_vectored_exception_handler_hook = std::make_unique<FunctionHookMinHook>(AddVectoredExceptionHandler, (uintptr_t)add_vectored_exception_handler_hook);
    if (!s_add_vectored_exception_handler_hook->create()) {
        spdlog::error("[IntegrityCheckBypass]: Could not hook AddVectoredExceptionHandler!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked AddVectoredExceptionHandler!");
}

PVOID WINAPI IntegrityCheckBypass::add_vectored_exception_handler_hook(ULONG FirstHandler, PVECTORED_EXCEPTION_HANDLER VectoredHandler) {
    s_veh_called = true;

    if (!s_veh_allowed) {
        const auto retaddr = (uintptr_t)_ReturnAddress();
        const auto module_within = utility::get_module_within(retaddr);

        if (module_within) {
            const auto module_path = utility::get_module_pathw(*module_within);
            bool is_allowed = false;

            if (module_path) {
                if (module_path->find(L"vehdebug") != std::wstring::npos) {
                    is_allowed = true;
                }

                if (module_path->find(L"coreclr") != std::wstring::npos) {
                    is_allowed = true;
                }

                if (module_path->find(L"dinput8") != std::wstring::npos) {
                    is_allowed = true;
                }
            }

            if (is_allowed || *module_within == REFramework::get_reframework_module()) 
            {
                if (module_path) {
                    spdlog::info("[IntegrityCheckBypass]: VEH allowed for {}", utility::narrow(*module_path));
                } else {
                    spdlog::info("[IntegrityCheckBypass]: VEH allowed");
                }

                return s_add_vectored_exception_handler_hook->get_original<decltype(add_vectored_exception_handler_hook)>()(FirstHandler, VectoredHandler);
            }
        }
        
        spdlog::warn("[IntegrityCheckBypass]: VEH not allowed, returning nullptr");
        allow_veh(); // VEH past this point should be okay.
        return (void*)VectoredHandler; // some bs address so it doesnt detect it as a nullptr
    }

    spdlog::info("[IntegrityCheckBypass]: VEH allowed");

    return s_add_vectored_exception_handler_hook->get_original<decltype(add_vectored_exception_handler_hook)>()(FirstHandler, VectoredHandler);
}

void IntegrityCheckBypass::hook_rtl_exit_user_process() {
    spdlog::info("[IntegrityCheckBypass]: Hooking RtlExitUserProcess...");

    const auto ntdll = GetModuleHandleW(L"ntdll.dll");

    if (ntdll == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find ntdll!");
        return;
    }

    const auto RtlExitUserProcess = GetProcAddress(ntdll, "RtlExitUserProcess");

    if (RtlExitUserProcess == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find RtlExitUserProcess!");
        return;
    }

    s_rtl_exit_user_process_hook = std::make_unique<FunctionHookMinHook>(RtlExitUserProcess, &rtl_exit_user_process_hook);
    if (!s_rtl_exit_user_process_hook->create()) {
        spdlog::error("[IntegrityCheckBypass]: Could not hook RtlExitUserProcess!");
        return;
    }

    spdlog::info("[IntegrityCheckBypass]: Hooked RtlExitUserProcess!");
}

void* IntegrityCheckBypass::rtl_exit_user_process_hook(uint32_t code) {
    /*__try {
        auto orig = s_rtl_exit_user_process_hook->get_original<decltype(rtl_exit_user_process_hook)>()(code);
        return orig;
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        TerminateProcess(GetCurrentProcess(), code);
        return nullptr;
    }*/

    spdlog::error("[IntegrityCheckBypass]: RtlExitUserProcess called with code 0x{:X} from 0x{:X}", code, (uintptr_t)_ReturnAddress());

    utility::exceptions::dump_callstack(nullptr);

    // ok for some reason I can't explain yet,
    // we need to do this because the game crashes if we don't
    // It seems to have something to do with RtlpFlsDataCleanup (which is called by RtlExitUserProcess)
    // and I think is responsible for calling the TLS destructors?
    // It calls something that's heap allocated but no longer exists, and it crashes.
    TerminateProcess(GetCurrentProcess(), code);
    return nullptr;
}

#pragma region Custom PAK directory loading

#define ENABLE_PAK_DIRECTORY_LOAD (TDB_VER >= 81)

static utility::ExhaustionResult do_exhaustion_scan_create_file_refs(utility::ExhaustionContext &ctx, uintptr_t target_search_func, std::vector<uintptr_t> &before_create_file_ptrs) {
    if (ctx.instrux.Category == ND_CAT_CALL) {
        if (ctx.instrux.Instruction == ND_INS_CALLNI) {
            auto displacement_opt = utility::resolve_displacement(ctx.addr);
            if (displacement_opt && *(uintptr_t*)(*displacement_opt) == target_search_func) {
                spdlog::info("[IntegrityCheckBypass]: Found stream open's call to CreateFileW at 0x{:X}, hooking it!", ctx.addr);
                before_create_file_ptrs.push_back(ctx.addr);
            }
        }

        return utility::ExhaustionResult::STEP_OVER;
    }

    return utility::ExhaustionResult::CONTINUE;
}

void IntegrityCheckBypass::find_try_hook_via_file_load_win32_create_file(uintptr_t pak_load_func_addr) {
#if ENABLE_PAK_DIRECTORY_LOAD
    // Find the first call instruction, thats our opening PAK file function
    const int INSTRUCTION_SEARCH_COUNT = 520;

    uint8_t *open_stream_func_addr = 0;
    uint8_t *search_current = (uint8_t*)pak_load_func_addr;

    uintptr_t last_call = 0;

    // The only clear indication of this function for now is that it is the first call function that checks its boolean result
    utility::linear_decode(search_current, INSTRUCTION_SEARCH_COUNT, [&](utility::ExhaustionContext& ctx) -> bool {
        auto &instr = ctx.instrux;
        if (instr.Instruction == ND_INS_CALLNR) {
            auto displacement_result = utility::resolve_displacement(ctx.addr);
            if (displacement_result) {
                last_call = *displacement_result;
            }
        }

        if (instr.Instruction == ND_INS_TEST) {
            if (instr.Operands[0].Type == ND_OP_REG && instr.Operands[0].Info.Register.Reg == NDR_RAX
                && instr.Operands[1].Type == ND_OP_REG && instr.Operands[1].Info.Register.Reg == NDR_RAX) {
                spdlog::info("[IntegrityCheckBypass]: Found call to stream open function at 0x{:X}!", (uintptr_t)last_call);
                open_stream_func_addr = (uint8_t*)last_call;
                return false;
            }
        }

        return true;
    });

    if (open_stream_func_addr == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find call to stream open function!");
        return;
    }

    uintptr_t target_search_func = (uintptr_t)&CreateFileW;
    const std::size_t exhaustive_decode_max = 12000;

    std::vector<uintptr_t> before_create_file_ptrs{};

    spdlog::info("[IntegrityCheckBypass]: Exhaustively decoding from 0x{:X} to find calls to CreateFileW...", (uintptr_t)open_stream_func_addr);

    utility::exhaustive_decode(open_stream_func_addr, exhaustive_decode_max, [target_search_func, &before_create_file_ptrs](utility::ExhaustionContext &ctx) {
        return do_exhaustion_scan_create_file_refs(ctx, target_search_func, before_create_file_ptrs);
    });

    for (auto ptr : before_create_file_ptrs) {
        auto hook = safetyhook::create_mid((void*)ptr, &IntegrityCheckBypass::via_file_prepare_to_create_file_w_hook_wrappper);
        if (hook) {
            spdlog::info("[IntegrityCheckBypass]: Successfully hooked instruction before CreateFileW at 0x{:X}!", ptr);
            s_before_create_file_w_hooks.push_back(std::move(hook));
        } else {
            spdlog::error("[IntegrityCheckBypass]: Failed to hook instruction before CreateFileW at 0x{:X}!", ptr);
        }
    }

    const char *direct_storage_open_pak_pattern[] = {
        "48 8D 56 08 48 8D 7C 24 ? 48 C7 07 00 00 00 00 48 8B 0D ? ? ? ? 48 8B 01 4C 8D 05 ? ? ? ? 49 89 F9 FF 50 20 48 8B 0F 85 C0", // MHWILDS v1041/MHSTORIES3
        "48 8D 56 08 48 8B 01 4C 8D 4D ? 4C 8D 05 ? ? ? ? FF 50 20 85 C0"   // Pragmata
    };

    std::optional<uintptr_t> direct_storage_open_pak_func_addr;

    for (const auto& pattern : direct_storage_open_pak_pattern) {
        auto addr = utility::scan(utility::get_executable(), pattern);
        if (addr) {
            direct_storage_open_pak_func_addr = addr;
            break;
        }
    }

    if (!direct_storage_open_pak_func_addr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find DirectStorage pak open block!");
        return;
    }

    // Find the first CALLNI instruction, which is the call to open the pak file stream
    uint8_t *direct_storage_before_open_pak_call = nullptr;
    search_current = (uint8_t*)*direct_storage_open_pak_func_addr;

    const int DIRECT_STORAGE_OPEN_PAK_CALL_SEARCH_COUNT = 25;

    for (int i = 0; i < DIRECT_STORAGE_OPEN_PAK_CALL_SEARCH_COUNT; i++) {
        auto instr = utility::decode_one(search_current);
        if (!instr) {
            continue;
        }

        if (instr->Instruction == ND_INS_CALLNI) {
            direct_storage_before_open_pak_call = search_current;
            break;
        }

        search_current += instr->Length;
    }

    if (direct_storage_before_open_pak_call == nullptr) {
        spdlog::error("[IntegrityCheckBypass]: Could not find call to open pak file stream in DirectStorage pak open block!");
        return;
    }

    s_directstorage_open_pak_hook = safetyhook::create_mid((void*)direct_storage_before_open_pak_call, &IntegrityCheckBypass::directstorage_open_pak_hook_wrappper);
    spdlog::info("[IntegrityCheckBypass]: Hooked DirectStorage pak open function at 0x{:X}!", (uintptr_t)direct_storage_before_open_pak_call);
#else
    spdlog::info("[IntegrityCheckBypass]: Custom pak directory loading is not supported for TDB version {}", TDB_VER);
#endif
}

int IntegrityCheckBypass::cache_and_count_custom_pak_in_directory() {
#if !ENABLE_PAK_DIRECTORY_LOAD
    return 0;
#else
    if (!m_load_pak_directory || !m_load_pak_directory->value()) {
        spdlog::info("[IntegrityCheckBypass]: Pak directory loading is disabled, skipping it.");
        return 0;
    }

    if (m_custom_pak_in_directory_paths_cached) {
        return static_cast<int>(m_custom_pak_in_directory_paths.size());
    }

    spdlog::info("[IntegrityCheckBypass]: Caching custom pak paths in executable directory...");
    m_custom_pak_in_directory_paths_cached = true;

    auto exe_module = utility::get_executable();
    auto exe_path = utility::get_module_pathw(exe_module);
    auto exe_dir = std::filesystem::path(*exe_path).parent_path();
    auto pak_dir_fs_path = exe_dir / CUSTOM_PAK_DIRECTORY_PATH;

    if (!std::filesystem::exists(pak_dir_fs_path)) {
        spdlog::warn("[IntegrityCheckBypass]: Custom pak directory does not exist at path: {}", utility::narrow(pak_dir_fs_path.wstring()));
        return 0;
    }

    // Iterate through the directory (recursively), and cache paths of all .pak files
    for (const auto& entry : std::filesystem::recursive_directory_iterator(pak_dir_fs_path)) {
        if (entry.is_regular_file() && entry.path().extension() == PAK_EXTENSION_NAME) {
            m_custom_pak_in_directory_paths.push_back(entry.path());
            spdlog::info("[IntegrityCheckBypass]: Cached custom pak with name: {} at path: {}", entry.path().filename().string(), utility::narrow(entry.path().wstring()));
        }
    }

    spdlog::info("[IntegrityCheckBypass]: Finished caching custom pak paths. Total count: {}", m_custom_pak_in_directory_paths.size());
    return static_cast<int>(m_custom_pak_in_directory_paths.size());
#endif
}

std::optional<int> IntegrityCheckBypass::extract_patch_num_from_path(std::wstring &path) {
    std::wsmatch match;
    if (std::regex_match(path, match, m_sub_patch_scan_regex)) {
        try {
            const int patch_num = std::stoi(match[1].str());
            return patch_num;
        } catch (const std::exception& e) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

void IntegrityCheckBypass::via_file_prepare_to_create_file_w_hook_wrappper(safetyhook::Context& context) {
    auto instance = IntegrityCheckBypass::get_shared_instance();
    if (instance) {
        instance->via_file_prepare_to_create_file_w_hook(context);
    } else {
        spdlog::error("[IntegrityCheckBypass]: Shared instance is null in via_file_prepare_to_create_file_w_hook_wrapper!");
    }
}

void IntegrityCheckBypass::directstorage_open_pak_hook_wrappper(safetyhook::Context& context) {
    auto instance = IntegrityCheckBypass::get_shared_instance();
    if (instance) {
        instance->directstorage_open_pak_hook(context);
    } else {
        spdlog::error("[IntegrityCheckBypass]: Shared instance is null in directstorage_open_pak_hook_wrapper!");
    }
}

void IntegrityCheckBypass::correct_pak_load_path(safetyhook::Context& context, int register_index) {
    static std::atomic<uint32_t> s_call_count{0};
    const auto call_index = s_call_count.fetch_add(1);
    if (call_index < 4) {
        auto* path_ptr = disasm_utils::get_register_value<wchar_t*>(context, register_index);
        const auto path_ok = path_ptr != nullptr && !IsBadStringPtrW(path_ptr, 1024);
        const std::string path_str = path_ok ? utility::narrow(path_ptr) : std::string{"<bad>"};
        spdlog::info("[IntegrityCheckBypass]: correct_pak_load_path #{} register {} path '{}'", call_index, register_index, path_str);

        // The trampoline the mid-hook landed in is exactly where dump_callstack() stops, so the
        // game frames below it -- the integrity check that decided to open the crash-report file --
        // are only reachable from the live RSP.
        log_stack_scan((uintptr_t)context.rsp, 0x100, "pak-open");

        if (call_index == 0) {
            utility::exceptions::dump_callstack(nullptr);
        }
    }

    if (!m_load_pak_directory || !m_load_pak_directory->value() || m_custom_pak_in_directory_paths.empty()) {
        return;
    }

    // Injected paks are handed to PakLoad under their bare native name, which deliberately does
    // not exist on disk, so divert the real open (CreateFileW / DirectStorage) to the actual file.
    // Keyed on the exact name rather than patch-number arithmetic, so it cannot drift out of sync
    // with what the injection loop synthesised.
    if (auto* path_ptr = disasm_utils::get_register_value<wchar_t*>(context, register_index);
        path_ptr != nullptr && !IsBadStringPtrW(path_ptr, 1024)) {
        if (const auto it = s_injected_name_to_real_path.find(path_ptr); it != s_injected_name_to_real_path.end()) {
            spdlog::info("[IntegrityCheckBypass]: redirecting open of '{}' -> '{}'",
                utility::narrow(it->first), utility::narrow(it->second));

            disasm_utils::set_register_value(context, register_index, it->second.c_str());
            return;
        }
    }
}

void IntegrityCheckBypass::directstorage_open_pak_hook(safetyhook::Context& context) {
    correct_pak_load_path(context, NDR_RDX);
}

void IntegrityCheckBypass::via_file_prepare_to_create_file_w_hook(safetyhook::Context& context) {
    correct_pak_load_path(context, NDR_RCX);
}

#pragma endregion

void IntegrityCheckBypass::on_config_load(const utility::Config& cfg) {
    for (IModValue& option : m_options) {
        option.config_load(cfg);
    }
}

void IntegrityCheckBypass::on_config_save(utility::Config& cfg) {
    for (IModValue& option : m_options) {
        option.config_save(cfg);
    }
}

void IntegrityCheckBypass::on_draw_ui() {
#if ENABLE_PAK_DIRECTORY_LOAD
    // In the universal build ENABLE_PAK_DIRECTORY_LOAD is always 1 (TDB_VER=84 >= 81),
    // so we must gate the UI at runtime for games that don't support PAK directory loading.
    if (sdk::GameIdentity::get().tdb_ver() < 81) {
        return;
    }
    if (!ImGui::CollapsingHeader("PAK Directory Loading")) {
        return;
    }

    ImGui::Text("Allow loading PAKs inside %s directory. PAKs can be of any filename and ends with .pak (case-sensitive)", IntegrityCheckBypass::CUSTOM_PAK_DIRECTORY_PATH);
    ImGui::Text("Restart the game to apply changes.");

    auto changed = false;
    changed |= m_load_pak_directory->draw("Enable");

    if (changed) {
        g_framework->request_save_config();
    }

    if (ImGui::TreeNode("List of custom PAKs loaded:")) {
        for (const auto& pak_path : m_custom_pak_in_directory_paths) {
            auto pak_utf8 = utility::narrow(pak_path);
            ImGui::BulletText("%s", pak_utf8.c_str());
        }
        ImGui::TreePop();
    }
#endif
}