#!/bin/bash
pkill -f "scripts/run_post_captur[e]"; sleep 1; pgrep -fa "scripts/run_post_captur[e]" | wc -l
