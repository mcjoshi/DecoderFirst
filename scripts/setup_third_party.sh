#!/usr/bin/env bash
# Clone the third-party code at the commits used for the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
clone() {
  [ -d "third_party/$1" ] || git clone "$2" "third_party/$1"
  git -C "third_party/$1" checkout "$3"
}
clone PretrainedSED https://github.com/fschmid56/PretrainedSED.git 1aa47e4
clone DESED_task https://github.com/DCASE-REPO/DESED_task.git c6bcb45
