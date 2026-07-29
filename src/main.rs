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
            let c_str = CStr::from_ptr(c_string);

            // 7. Convert to a standard Rust String!
            let final_rust_string = c_str.to_str().unwrap().to_owned();



            vlog!("Successfully grabbed text from macOS:");
            vlog!("-> {}\n", final_rust_string);
            // Let's test your new splitter!
                        let sentence_queue = split_line_form_para(&final_rust_string);
                        vlog!("🔪 Sliced into {} sentences:", sentence_queue.len());
                        thread::spawn(move || {
                             use std::io::Write;
                                        vlog!("🤖 Booting Native Swift Engine...");

                                        // Start the compiled Swift binary
                                        // ... existing code ...
                                                        // Start the compiled Swift binary
                                                        // model + metallib paths now handled by VocalWorker (VOCAL_MODEL_PATH + current_dir)

                                                                      // Start the compiled Swift binary
                                                                      let mut child = Command::new("/Users/singupallikartik/Developer/fun-projects/swift-qwen3-tts/.build/release/VocalWorker")
                                            .current_dir("/Users/singupallikartik/Developer/fun-projects/swift-qwen3-tts")
                                            .env("VOCAL_MODEL_PATH", "/Users/singupallikartik/Developer/fun-projects/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit").env("VOCAL_SPEAKER", "Dylan").env("VOCAL_LANGUAGE", "English").env("VOCAL_INSTRUCT", "be very Fast, Serious, and not skip any word like you are reading audiobook").env("VOCAL_TEMPERATURE", "0.8")
                                                                          .stdin(Stdio::piped()).stdout(Stdio::piped())
                                                                          // metallib is found via current_dir (swift-qwen3-tts/default.metallib)
                                                                          .spawn()
                                                                          .expect("Failed to start Native Swift Engine");
                                        // ... existing code ...

                                        // Take the standard input pipe
                                        let mut stdin = child.stdin.take().expect("Failed to open stdin");

                                        // Capture the Swift worker's stdout and tee it into our log
                                        // file too, so its "Generating native audio for: ..." lines
                                        // are visible via `tail -f /tmp/vocal.log` without editing
                                        // the swift-qwen3-tts repo.
                                        let child_stdout = child.stdout.take();
                                        let swift_reader = thread::spawn(move || {
                                            use std::io::BufRead;
                                            if let Some(out) = child_stdout {
                                                for line in std::io::BufReader::new(out)
                                                    .lines()
                                                    .map_while(Result::ok)
                                                {
                                                    vlog!("[swift] {}", line);
                                                }
                                            }
                                        });

                                        // Stream sentences to Swift one by one
                                        for (i, sentence) in sentence_queue.iter().enumerate() {
                                            vlog!("  🔊 Streaming sentence {} to Native Worker...", i + 1);
                                            writeln!(stdin, "{}", sentence).expect("Failed to write to native worker");
                                        }

                                        // Drop stdin! This tells Swift "We are done reading, shut down!"
                                        drop(stdin);

                                        // Wait for Swift to finish generating audio and gracefully shut down
                                        let _ = child.wait().expect("Failed to wait on child");
                                        let _ = swift_reader.join();

                                        vlog!("🛑 Finished reading. Swift engine killed and RAM freed!");
                                    });
        }
    }
);
fn split_line_form_para(data: &String) -> Vec<String> {
    data.split_terminator(&['.', '?', '!'][..])
        .map(|s| s.trim().to_string()) // Convert to String and strip extra spaces
        .filter(|s| !s.is_empty()) // Ignore empty chunks
        .collect()
}
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
