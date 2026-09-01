#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

use objc2::runtime::{AnyObject, NSObject};
use objc2::{ClassType, DeclaredClass, class, declare_class, msg_send, mutability};
use objc2_foundation::NSString;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use vocal::VocalConfig;

/// Win-win logging: writes each line to stdout (so a terminal-launched run
/// still shows it inline) AND appends it to /tmp/vocal.log. That file is the
/// key — even when macOS spawns its own terminal-less instance to handle a
/// right-click (whose stdout is /dev/null), every log still lands in the file.
/// View it live with:  tail -f /tmp/vocal.log
macro_rules! vlog {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        {
            use std::io::Write;
            let _ = writeln!(std::io::stdout(), "{}", line);
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/vocal.log")
            {
                let _ = writeln!(f, "{}", line);
            }
        }
    }};
}

// flock(2) operations on macOS (no libc crate needed — std links libSystem).
const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;

unsafe extern "C" {
    /// BSD flock(2) — backs the single-host lock (/tmp/vocal.host.lock).
    fn flock(fd: c_int, operation: c_int) -> c_int;
}

/// Set the moment macOS delivers ANY service request to us. A launch that was
/// triggered by right-click → Speak with Vocal gets its request almost
/// immediately, while a user double-click on Vocal.app gets nothing — so the
/// delayed Manager-open checks this flag first and stays quiet for
/// service-triggered launches (no GUI popping up on every right-click).
static SERVICE_REQUEST_SEEN: AtomicBool = AtomicBool::new(false);

/// Path of the Vocal Manager app embedded in our bundle
/// (`Contents/Managers/*.app`), or None in dev mode / when not embedded.
fn manager_app_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    if !exe.to_string_lossy().contains(".app/Contents/MacOS/") {
        return None;
    }
    let managers = exe.parent()?.parent()?.join("Managers");
    std::fs::read_dir(managers)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|ext| ext == "app"))
}

/// Open the Vocal Manager GUI (a no-op in dev mode / if not embedded).
/// `open` activates the Manager if it's already running instead of forking a
/// second copy — double-clicking Vocal.app always lands on one window.
fn open_manager() {
    match manager_app_path() {
        Some(path) => match Command::new("/usr/bin/open").arg(&path).spawn() {
            Ok(_) => vlog!("🚀 Opened Vocal Manager: {}", path.display()),
            Err(e) => vlog!("⚠️ Could not open Vocal Manager: {e}"),
        },
        None => vlog!("(no embedded Vocal Manager to open — dev mode?)"),
    }
}

/// Try to become THE background host. Returns Err when another Vocal host is
/// already running (it holds the lock) — in that case this launch came from a
/// user double-click on Vocal.app, and its only job is to show the Manager.
fn try_become_host() -> Result<(), ()> {
    use std::os::unix::io::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open("/tmp/vocal.host.lock")
        .map_err(|_| ())?;
    if unsafe { flock(lock.as_raw_fd(), LOCK_EX | LOCK_NB) } != 0 {
        return Err(()); // another host holds the lock
    }
    // Leak the File so the fd (and therefore the lock) lives as long as we do.
    std::mem::forget(lock);
    Ok(())
}

// 1. Define our Objective-C Class using the modern objc2 macro
declare_class!(
    struct VocalService;

    unsafe impl ClassType for VocalService {
        type Super = NSObject;
        type Mutability = mutability::InteriorMutable;
        const NAME: &'static str = "VocalService";
    }

    impl DeclaredClass for VocalService {}

    unsafe impl VocalService {
        // 2. The Method macOS calls when right-clicked
        #[method(handleSpeakText:userData:error:)]
        unsafe fn handle_speak_text(
            &self,
            pboard: *mut AnyObject,      // The Pasteboard
            _user_data: *mut AnyObject,  // Unused
            _error: *mut *mut AnyObject, // Unused
        ) {
            // Marks this launch as service-triggered (cancels the delayed
            // Manager-open — see main()).
            SERVICE_REQUEST_SEEN.store(true, Ordering::SeqCst);

            vlog!("\n🔔 macOS just triggered our Rust app!");

            // 3. Create an Apple String for "public.utf8-plain-text"
            let utf8_type = NSString::from_str("public.utf8-plain-text");

            // 4. Ask the pasteboard for the text
            let ns_text: *mut AnyObject = msg_send![pboard, stringForType: &*utf8_type];

            if ns_text.is_null() {
                vlog!("Pasteboard didn't contain text.");
                return;
            }

            // 5. Convert Apple string pointer to C string pointer
            let c_string: *const c_char = msg_send![ns_text, UTF8String];

            // 6. Safely wrap it in a Rust CStr
            let c_str = unsafe { CStr::from_ptr(c_string) };

            // 7. Convert to a standard Rust String!
            let final_rust_string = c_str.to_str().unwrap().to_owned();

            vlog!("Successfully grabbed text from macOS:");
            vlog!("-> {}\n", final_rust_string);

            let sentence_queue = vocal::split_paragraph(&final_rust_string);
            vlog!("🔪 Sliced into {} sentences:", sentence_queue.len());

            thread::spawn(move || {
                use std::io::Write;

                // Single source of truth: vocal.config (gitignored) + baked-in defaults.
                let cfg = VocalConfig::load_default();
                vlog!("🤖 Booting TTS worker (backend={})...", cfg.backend);

                // Build the worker Command from the active backend (swift | chatterbox).
                let spec = vocal::worker::launch_spec(&cfg);
                let mut cmd = Command::new(&spec.program);
                cmd.args(&spec.args);
                for (k, v) in &spec.envs {
                    cmd.env(k, v);
                }
                if let Some(cwd) = &spec.cwd {
                    cmd.current_dir(cwd);
                }
                let mut child = cmd
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .expect("Failed to start TTS worker");

                let mut stdin = child.stdin.take().expect("Failed to open stdin");

                // Capture the worker's stdout and tee it into the log file.
                let child_stdout = child.stdout.take();
                let worker_reader = thread::spawn(move || {
                    use std::io::BufRead;
                    if let Some(out) = child_stdout {
                        for line in std::io::BufReader::new(out)
                            .lines()
                            .map_while(Result::ok)
                        {
                            vlog!("[worker] {}", line);
                        }
                    }
                });

                // Stream sentences to the worker one by one (one per line).
                // A failed write means the worker is gone — the user hit
                // Stop Vocal (or it crashed) — so drop the rest of the queue
                // instead of panicking the thread.
                for (i, sentence) in sentence_queue.iter().enumerate() {
                    vlog!("  🔊 Streaming sentence {} to worker...", i + 1);
                    if let Err(e) = writeln!(stdin, "{}", sentence) {
                        vlog!(
                            "  ⛔ Worker is gone ({e}) — not streaming the remaining {} sentence(s).",
                            sentence_queue.len() - i - 1
                        );
                        break;
                    }
                }

                // Drop stdin! This tells the worker "we are done, shut down".
                drop(stdin);

                let _ = child.wait().expect("Failed to wait on child");
                let _ = worker_reader.join();

                vlog!("🛑 Finished reading. Worker killed and RAM freed!");
            });
        }

        // Second Service: right-click any text → Services → Stop Vocal.
        // Kills the running TTS worker so playback halts instantly — no more
        // Activity Monitor when a paragraph turns out too long to hear out.
        // The pattern is anchored to the binary path (`/ChatterboxMLWorker$`)
        // so it matches only real worker processes (bundle- OR dev-spawned,
        // host or MCP) — never a terminal/grep/editor that merely mentions
        // the name. Non-zero pkill exit = nothing was playing.
        #[method(handleStopSpeech:userData:error:)]
        unsafe fn handle_stop_speech(
            &self,
            _pboard: *mut AnyObject,     // The Pasteboard (ignored)
            _user_data: *mut AnyObject,  // Unused
            _error: *mut *mut AnyObject, // Unused
        ) {
            vlog!("\n🛑 Stop Vocal service triggered!");
            match Command::new("/usr/bin/pkill")
                .arg("-f")
                .arg("/ChatterboxMLWorker$")
                .status()
            {
                Ok(status) if status.success() => {
                    vlog!("🛑 TTS worker stopped — audio halted, RAM + GPU freed.");
                }
                _ => vlog!("💤 Nothing was playing — no worker to stop."),
            }
        }
    }
);

// App delegate: Finder double-click while the host is already running sends a
// reopen event to the EXISTING instance (LaunchServices routes by bundle id) —
// treat that exactly like a fresh click and show the Manager.
declare_class!(
    struct AppDelegate;

    unsafe impl ClassType for AppDelegate {
        type Super = NSObject;
        type Mutability = mutability::InteriorMutable;
        const NAME: &'static str = "VocalAppDelegate";
    }

    impl DeclaredClass for AppDelegate {}

    unsafe impl AppDelegate {
        #[method(applicationShouldHandleReopen:hasVisibleWindows:)]
        unsafe fn should_handle_reopen(
            &self,
            _app: *mut AnyObject,
            _has_visible_windows: bool, // BOOL == bool on arm64 macOS
        ) -> bool {
            vlog!("🖱️ Vocal.app re-opened — showing the Manager.");
            open_manager();
            true
        }
    }
);

fn main() {
    vlog!(
        "=== Vocal started (pid {}, logs -> /tmp/vocal.log) ===",
        std::process::id()
    );

    // Single-host lock: if a host is already running, this launch is a user
    // double-click on Vocal.app — just show the Manager and exit.
    if try_become_host().is_err() {
        vlog!("👋 Host already running — opening the Manager for this click.");
        open_manager();
        return;
    }

    vlog!("🚀 Starting Vocal background service...");

    // Boot diagnostic: log where assets resolve from. In a .app bundle these
    // point at Contents/Resources/; in dev (`cargo run`) at the repo checkout.
    // Confirms the current_exe-based bundle detection works at runtime.
    let boot = VocalConfig::load_default();
    vlog!(
        "📦 resolving from: worker={}, cwd={}, model={}",
        boot.chatterbox_binary().display(),
        boot.engine_cwd().display(),
        boot.native_model_dir()
    );

    unsafe {
        // 1. Get the shared NSApplication instance (this represents our Mac app)
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];

        // 2. Create an instance of our custom VocalService
        let service: *mut AnyObject = msg_send![VocalService::class(), new];

        // 3. Tell macOS that this object handles our right-click services
        let _: () = msg_send![app, setServicesProvider: service];

        // 4. Reopen hook: clicking Vocal.app while we're the running host.
        let delegate: *mut AnyObject = msg_send![AppDelegate::class(), new];
        let _: () = msg_send![app, setDelegate: delegate];

        vlog!("✅ Vocal is listening for right-clicks (Speak with Vocal / Stop Vocal).");

        // 5. Click vs. service-request disambiguation: a launch triggered by
        // right-click → Speak with Vocal delivers its request within ~100ms,
        // while a user double-click on Vocal.app delivers nothing. So open the
        // Manager only if no service request has arrived after a short grace
        // period — GUI stays out of the way on every right-click.
        thread::spawn(|| {
            thread::sleep(Duration::from_millis(900));
            if !SERVICE_REQUEST_SEEN.load(Ordering::SeqCst) {
                open_manager();
            }
        });

        // 6. Start the app's event loop so it stays alive in the background
        let _: () = msg_send![app, run];
    }
}
