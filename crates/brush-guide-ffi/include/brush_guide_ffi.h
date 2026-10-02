// C interface of crates/brush-guide-ffi: one live guidance session.
// Frames are the wire frames of docs/protocol.md.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct BgeEngine BgeEngine;

// Receives server frames; the frame is valid only during the call.
// Threading: out is called on engine threads and also synchronously on the
// calling thread inside bge_new, bge_push and bge_finish (ack, error, splat),
// so calls can overlap; out and ctx must tolerate concurrent calls from
// several threads. No bge_* function may be called from inside out. Ordering
// across threads is not guaranteed: the ack of keyframe N can arrive after a
// score set that already includes it.
typedef void (*BgeOut)(void *ctx, const uint8_t *frame, size_t len);

// config_json: GuideConfig overrides (may be ""). Returns NULL on failure after
// passing one error frame to out.
BgeEngine *bge_new(const char *config_json, const char *session_dir, BgeOut out, void *ctx);
// Blocks until the keyframe is added (ack emitted) or rejected (error emitted); waits while paused.
int32_t bge_push(BgeEngine *engine, const uint8_t *frame, size_t len);
// Writes the splat PLY and emits a splat frame.
int32_t bge_finish(BgeEngine *engine, const char *ply_path);
void bge_reset(BgeEngine *engine);
// Returns once this engine runs no GPU work (its own warm-up pauses between sizes).
// A bge_warm_up running at the same time on another thread is not paused.
void bge_pause(BgeEngine *engine);
void bge_resume(BgeEngine *engine);
// Runs the session warm-up (GPU autotuning at the configured splat budget and keyframe size) and
// returns when it is done: 0 on success, 1 on a config error, no GPU or a failed warm-up. Takes the
// same config JSON as bge_new (including gpu_autotune_level / gpu_autotune_samples) and uses the same
// process-wide GPU device, so a later bge_new reuses the tuning. Blocking: call it from a background
// thread, not from inside out. May run while a capture's bge_new / bge_push are running; the device
// setup is shared and guarded.
int32_t bge_warm_up(const char *config_json);
// Stops the engine; out is not called after it returns. No other bge_* call may
// be in progress on any thread.
void bge_free(BgeEngine *engine);
