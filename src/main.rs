#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

use objc2::runtime::{AnyObject, NSObject};
use objc2::{ClassType, DeclaredClass, class, declare_class, msg_send, mutability};
use objc2_foundation::NSString;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::process::{Command, Stdio};
use std::thread;

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
                let cfg = VocalConfig::load(std::path::Path::new("vocal.config"));
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
                for (i, sentence) in sentence_queue.iter().enumerate() {
                    vlog!("  🔊 Streaming sentence {} to worker...", i + 1);
                    writeln!(stdin, "{}", sentence).expect("Failed to write to worker");
                }

                // Drop stdin! This tells the worker "we are done, shut down".
                drop(stdin);

                let _ = child.wait().expect("Failed to wait on child");
                let _ = worker_reader.join();

                vlog!("🛑 Finished reading. Worker killed and RAM freed!");
            });
        }
    }
);

fn main() {
    vlog!(
        "=== Vocal started (pid {}, logs -> /tmp/vocal.log) ===",
        std::process::id()
    );
    vlog!("🚀 Starting Vocal background service...");

    unsafe {
        // 1. Get the shared NSApplication instance (this represents our Mac app)
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];

        // 2. Create an instance of our custom VocalService
        let service: *mut AnyObject = msg_send![VocalService::class(), new];

        // 3. Tell macOS that this object handles our right-click services
        let _: () = msg_send![app, setServicesProvider: service];

        vlog!("✅ Vocal is now listening for right-clicks! (Keep this terminal open)");

        // 4. Start the app's event loop so it stays alive in the background
        let _: () = msg_send![app, run];
    }
}
