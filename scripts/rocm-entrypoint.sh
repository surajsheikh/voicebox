#!/bin/sh
set -e
# Join whatever groups own the mounted GPU nodes so /dev/kfd and /dev/dri work
# on any host (no RENDER_GID/VIDEO_GID needed), then drop to the app user.
for dev in /dev/kfd /dev/dri/render*; do
    [ -e "$dev" ] || continue
    gid=$(stat -c %g "$dev")
    grp=$(getent group "$gid" | cut -d: -f1)
    [ -n "$grp" ] || {
        grp="gpu$gid"
        groupadd -g "$gid" "$grp"
    }
    usermod -aG "$grp" voicebox
done
# The model cache (voicebox_models volume, mounted at /home/voicebox/.cache)
# persists across container recreates, so any root-owned subdirectory left
# behind by an earlier root-run process sticks around indefinitely — the
# unprivileged `voicebox` user below can then read the existing model dirs
# fine but gets a PermissionError trying to create a NEW one (confirmed live:
# blocked downloading the 1.7B model while 0.6B's dir, created earlier, was
# already there and fine). Self-heal on every start rather than relying on a
# one-off manual chown that doesn't survive a fresh host or volume recreate.
chown -R voicebox:voicebox /home/voicebox/.cache 2>/dev/null || true
exec gosu voicebox "$@"
