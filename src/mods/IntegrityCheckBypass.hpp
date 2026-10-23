#pragma once

#include <memory>
#include <string_view>
#include <regex>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "Mod.hpp"
#include "utility/Patch.hpp"
#include "utility/FunctionHook.hpp"
#include "utility/FunctionHookMinHook.hpp"

#include <bddisasm.h>

// Always on for RE3
// Because we use hooks that modify the integrity of the executable
// And RE3 has unfortunately decided to implement an integrity check on the executable code of the process
class IntegrityCheckBypass : public Mod {
public:
    std::string_view get_name() const override { return "IntegrityCheckBypass"; };
    std::optional<std::string> on_initialize() override;

    void on_frame() override;
    void on_config_load(const utility::Config& cfg) override;
    void on_config_save(utility::Config& cfg) override;
    void on_draw_ui() override;
    
    static void ignore_application_entries();
    static void immediate_patch_re8();
    static void immediate_patch_re4();
    static void immediate_patch_dd2();
    static void immediate_patch_re9();
    static void re9_heartbeat_bypass();
    static void remove_stack_destroyer();

    // Must run from startup_thread BEFORE the REFramework constructor: the crash frame is already
    // built by the time remove_stack_destroyer() runs.
    static void early_mhwilds_diagnostics();

    // Reads the bisect switches straight off re2_fw_config.txt during the early phase. Must be a
    // member: the switches are private static members, so a free function cannot name them.
    static void load_bisect_config();

    // ---- early group switches -------------------------------------------------------------------
    // Group switches for the parts of REFramework that are NOT integrity-bypass patches: the module
    // spoof, the always-on hooks, the exception handler, the mods and the always-on diagnostics.
    // Registered options like the patches, but they have to be resolved at the very start of DllMain
    // -- long before the normal mod config load -- so load_early_switches() reads them there.
    static void load_early_switches();
    static bool diagnostics_disabled();
    static bool module_spoof_disabled();
    static bool hooks_disabled();
    static bool exception_handler_disabled();
    static bool mods_disabled();
    // FaultyFileDetector has no ModToggle of its own, so it gets a switch here.
    static bool faulty_file_detector_disabled();

    static void setup_pristine_syscall();
    static void fix_virtual_protect();

    static void hook_add_vectored_exception_handler();
    static void hook_rtl_exit_user_process();

    static void allow_veh() {
        s_veh_allowed = true;
    }
    
    static bool is_veh_called() {
        return s_veh_called;
    }

    static std::shared_ptr<IntegrityCheckBypass>& get_shared_instance();

private:
    static void* renderer_create_blas_hook(void* a1, void* a2, void* a3, void* a4, void* a5);
    static inline std::unique_ptr<FunctionHook> s_renderer_create_blas_hook{};
    static inline uint32_t* s_corruption_when_zero{ nullptr };
    static inline uint32_t s_last_non_zero_corruption{ 8 }; // What I've seen it default to

    static void sha3_rsa_code_midhook(safetyhook::Context& context);
    static bool pak_load_check_function(void* pak_struct, const wchar_t* pak_name, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7);
    static void* pak_load_patch_load_function(uintptr_t* pak_slots, const wchar_t* base_path, int32_t first_slot_index, int32_t load_flags);
    static int scan_patch_files_count();
    static void restore_unencrypted_paks();
    static inline safetyhook::MidHook s_sha3_rsa_code_midhook;
    static inline safetyhook::InlineHook s_pak_load_check_function_hook;
    static inline safetyhook::InlineHook s_pak_load_patch_load_hook;
    static inline std::optional<uintptr_t> s_sha3_code_end{};
    static inline int s_patch_count;
    static inline bool s_patch_count_checked;
    static inline std::optional<std::uint8_t> s_pak_flags_value{};
    static inline safetyhook::MidHook s_patch_store_flags_hook;
    static inline INSTRUX s_pak_load_check_insn{};

    static void anti_debug_watcher();
    static void init_anti_debug_watcher();
    static void nuke_heap_allocated_code(uintptr_t addr);
    static inline std::unique_ptr<std::jthread> s_anti_anti_debug_thread{nullptr};

    static BOOL WINAPI virtual_protect_impl(LPVOID lpAddress, SIZE_T dwSize, DWORD flNewProtect, PDWORD lpflOldProtect);
    static BOOL WINAPI virtual_protect_hook(LPVOID lpAddress, SIZE_T dwSize, DWORD flNewProtect, PDWORD lpflOldProtect);
    
    static PVOID WINAPI add_vectored_exception_handler_hook(ULONG FirstHandler, PVECTORED_EXCEPTION_HANDLER VectoredHandler);
    static inline bool s_veh_allowed{false};
    static inline bool s_veh_called{false};
    using NtProtectVirtualMemory_t =  NTSTATUS(NTAPI*)(HANDLE ProcessHandle, PVOID* BaseAddress, SIZE_T* NumberOfBytesToProtect, ULONG NewAccessProtection, PULONG OldAccessProtection);
    static inline NtProtectVirtualMemory_t s_pristine_protect_virtual_memory{ nullptr };
    static inline NtProtectVirtualMemory_t s_og_protect_virtual_memory{ nullptr };;

    // Using minhook because safetyhook crashes on trying to hook VirtualProtect
    static inline std::unique_ptr<FunctionHookMinHook> s_virtual_protect_hook{};
    static inline std::unique_ptr<FunctionHookMinHook> s_add_vectored_exception_handler_hook{};

    static void* rtl_exit_user_process_hook(uint32_t code);
    static inline std::unique_ptr<FunctionHookMinHook> s_rtl_exit_user_process_hook{};

#ifdef REFRAMEWORK_UNIVERSAL
    // All members present in monolithic build
    bool* m_bypass_integrity_checks{ nullptr };
    void disable_update_timers(std::string_view name) const;
    std::vector<std::unique_ptr<Patch>> m_patches{};
#else
#ifdef RE3
    // This is what the game uses to bypass its integrity checks altogether or something
    bool* m_bypass_integrity_checks{ nullptr };
#else
    void disable_update_timers(std::string_view name) const;

    std::vector<std::unique_ptr<Patch>> m_patches{};
#endif
#endif

#pragma region Custom PAK directory loading
    constexpr static const char* CUSTOM_PAK_DIRECTORY_PATH = "pak_mods";
    constexpr static const char* PAK_EXTENSION_NAME = ".pak";
    constexpr static const wchar_t* PAK_EXTENSION_NAME_W = L".pak";
    constexpr static const wchar_t* SUB_PATCH_SCAN_REGEX = LR"(re_chunk_000\.pak\.sub_000\.pak\.patch_(\d+)\.pak)";

    static void find_try_hook_via_file_load_win32_create_file(uintptr_t pak_load_func_addr);
    static void via_file_prepare_to_create_file_w_hook_wrappper(safetyhook::Context& context);
    static void directstorage_open_pak_hook_wrappper(safetyhook::Context& context);

    int cache_and_count_custom_pak_in_directory();
    std::optional<int> extract_patch_num_from_path(std::wstring &path);
    void via_file_prepare_to_create_file_w_hook(safetyhook::Context& context);
    void directstorage_open_pak_hook(safetyhook::Context& context);
    void correct_pak_load_path(safetyhook::Context& context, int register_index);

    const ModToggle::Ptr m_load_pak_directory{ ModToggle::create(generate_name("LoadPakDirectory"), true) };

    // Bisect switches for the integrity bypass. Every one defaults to the behaviour the game needs;
    // turning one off is a diagnosis tool, not a mode. Keys are IntegrityCheckBypass_<Name> in
    // re2_fw_config.txt. `DisableAllPatches` skips every bypass patch while leaving all diagnostics
    // (VEH, watchers, early phase) running, so "does the crash depend on our patches at all?" is a
    // single re-run rather than a rebuild.
    static inline const ModToggle::Ptr m_disable_all_patches{ ModToggle::create("IntegrityCheckBypass_DisableAllPatches", false) };
    static inline const ModToggle::Ptr m_patch_crash_report_check{ ModToggle::create("IntegrityCheckBypass_PatchCrashReportCheck", true) };
    // Instead of forcing the "file missing" branch (which skips launching CrashReport.exe entirely),
    // let the reporter run and force the outcome to "continue". Both paths converge on the same
    // continuation, so the only difference is whether the reporter actually ran first.
    static inline const ModToggle::Ptr m_crash_report_force_continue{ ModToggle::create("IntegrityCheckBypass_CrashReportCheckForceContinue", false) };
    static inline const ModToggle::Ptr m_patch_scanner_crasher{ ModToggle::create("IntegrityCheckBypass_PatchScannerCrasher", true) };
    static inline const ModToggle::Ptr m_hook_create_blas{ ModToggle::create("IntegrityCheckBypass_HookCreateBLAS", true) };
    static inline const ModToggle::Ptr m_patch_sus_constants{ ModToggle::create("IntegrityCheckBypass_PatchSusConstants", true) };
    static inline const ModToggle::Ptr m_patch_pak_integrity{ ModToggle::create("IntegrityCheckBypass_PatchPakIntegrity", true) };
    static inline const ModToggle::Ptr m_patch_stack_destroyer{ ModToggle::create("IntegrityCheckBypass_PatchStackDestroyer", true) };

    // Group switches for the features that REFramework installs from DllMain or the startup thread
    // before mod configs are loaded. Ordinary registered options -- save_config() preserves them and
    // the UI shows them -- but they have to be resolved early, which is what the prefix marks.
    // Defaults are the shipping behaviour: diagnostics off, everything else on. Turning one on (or
    // diagnostics on) is a diagnosis tool, not a mode.
    static inline const ModToggle::Ptr m_early_minimal{ ModToggle::create("Early_Minimal", false) };
    static inline const ModToggle::Ptr m_early_disable_diagnostics{ ModToggle::create("Early_DisableDiagnostics", true) };
    static inline const ModToggle::Ptr m_early_disable_module_spoof{ ModToggle::create("Early_DisableModuleSpoof", false) };
    static inline const ModToggle::Ptr m_early_disable_hooks{ ModToggle::create("Early_DisableHooks", false) };
    static inline const ModToggle::Ptr m_early_disable_exception_handler{ ModToggle::create("Early_DisableExceptionHandler", false) };
    static inline const ModToggle::Ptr m_early_disable_mods{ ModToggle::create("Early_DisableMods", false) };
    static inline const ModToggle::Ptr m_early_disable_faulty_file_detector{ ModToggle::create("Early_DisableFaultyFileDetector", false) };

    static inline std::vector<safetyhook::MidHook> s_before_create_file_w_hooks{};
    static inline safetyhook::MidHook s_directstorage_open_pak_hook{};
    static inline int s_base_directory_patch_count{0};

    constexpr static size_t PRISTINE_PAK_STRUCT_SIZE = 0x300;
    static inline std::array<uint8_t, PRISTINE_PAK_STRUCT_SIZE> s_pristine_pak_struct{};

    struct PakRebase {
        size_t offset;
        size_t delta_from_base;
    };

    static inline std::vector<PakRebase> s_pak_rebase_offsets{};
    static inline uintptr_t* s_pak_array_start{nullptr};
    static inline size_t s_pak_array_len{0};
    static inline size_t s_event_handle_offset{0};
    static inline size_t s_event_handle_offset_2{0};

    static inline std::unordered_map<std::wstring, std::wstring> s_injected_name_to_real_path{};

    static inline bool s_auto_assigned{false};
    static inline std::unordered_set<std::wstring> s_seen_pak_families{};

    std::vector<std::wstring> m_custom_pak_in_directory_paths{};
    bool m_custom_pak_in_directory_paths_cached{ false };
    std::wregex m_sub_patch_scan_regex{SUB_PATCH_SCAN_REGEX, std::regex::ECMAScript};

    ValueList m_options{
        *m_load_pak_directory,
        *m_disable_all_patches,
        *m_patch_crash_report_check,
        *m_crash_report_force_continue,
        *m_patch_scanner_crasher,
        *m_hook_create_blas,
        *m_patch_sus_constants,
        *m_patch_pak_integrity,
        *m_patch_stack_destroyer,
        *m_early_minimal,
        *m_early_disable_diagnostics,
        *m_early_disable_module_spoof,
        *m_early_disable_hooks,
        *m_early_disable_exception_handler,
        *m_early_disable_mods,
        *m_early_disable_faulty_file_detector
    };
#pragma endregion 
};