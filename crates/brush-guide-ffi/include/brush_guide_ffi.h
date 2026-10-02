// C interface of crates/brush-guide-ffi: one live guidance session.
// Frames are the wire frames of docs/protocol.md.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct BgeEngine BgeEngine;

// Called on engine threads with a server frame, valid only during the call.
typedef void (*BgeOut)(void *ctx, const uint8_t *frame, size_t len);

// config_json: GuideConfig overrides (may be ""). Returns NULL on failure after
// passing one error frame to out.
BgeEngine *bge_new(const char *config_json, const char *session_dir, BgeOut out, void *ctx);
// Blocks until the keyframe is added (ack emitted) or rejected (error emitted); waits while paused.
int32_t bge_push(BgeEngine *engine, const uint8_t *frame, size_t len);
// Writes the splat PLY and emits a splat frame.
int32_t bge_finish(BgeEngine *engine, const char *ply_path);
void bge_reset(BgeEngine *engine);
// Returns once no GPU work is running; must not be called on the thread blocked in bge_push.
void bge_pause(BgeEngine *engine);
void bge_resume(BgeEngine *engine);
void bge_free(BgeEngine *engine);
