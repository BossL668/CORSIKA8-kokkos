#!/usr/bin/env bash
exec /snap/bin/chromium --disable-gpu --disable-dev-shm-usage "$@"
