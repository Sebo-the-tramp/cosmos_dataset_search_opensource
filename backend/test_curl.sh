#!/usr/bin/env bash
set -euo pipefail

curl -s "http://127.0.0.1:5000/search?word=stroller&quantity=5" | python -m json.tool
