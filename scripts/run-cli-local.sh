#!/bin/sh
#

# Use relative path from TINYMOE_DIR
TINYMOE_DIR="${TINYMOE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
basedir="${LLAMA_CPP_DIR:-$TINYMOE_DIR/external/llama.cpp}"
gguf_dir="$TINYMOE_DIR/gguf"

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="models-q4_0.gguf"
[ "$M" != "" ] && model="$M"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

experimental=
[ "$E" != "" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

set -x

cd $basedir; ulimit -c unlimited;        \
  LD_LIBRARY_PATH=$basedir/$branch/lib   \
  ADSP_LIBRARY_PATH=$basedir/$branch/lib \
  $verbose $experimental $sched $opmask $profile $nhvx $ndev           \
    ./$branch/build/bin/llama-cli -m $gguf_dir/$model       \
        -t 16 --mlock --ctx-size 1024 --batch-size 1 --temp 0.5 --top_p 0.7 --seed 42 --no-display-prompt -fa off \
        --ignore-eos \
        -ngl 99 $cli_opts "$@" \

