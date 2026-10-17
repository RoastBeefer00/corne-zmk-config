#!/bin/bash
# Generate candidate layouts per thumb configuration: soft PM rule (HINGE=200) and no rule at all.
cd "$(dirname "$0")"; export CORPUS=${CORPUS:-corpus}
PM="vwgmj;.'/=snthk,aeicfpdlx-uoybr"
E5="xcgkv=;-,.nsthm'aeipbwdlf/ouyjr"
gen() { # name env fixed seed
  for mode in soft free; do
    if [ $mode = soft ]; then base=$PM; else base=-; fi
    env $2 HINGE=200 TOPN=24 ./target/release/kb opt 0 2000000 24 "$base" 0.02 "$3" "$4" 2>/dev/null \
      | awk -v n="$1/$mode" '/^cost/ {print n "\t" $3$4$5$6}'
  done
}
gen R "X=1" r "$E5"
gen E-vowels-right "VOWELS=right" e "xcgkv=;-,.nsthm'arip bwdlf/ouyje"
gen E-vowels-left "VOWELS=left" e "jpou-fgdl=yhai.bstnr/k',;vcmwxe"
gen free "X=1" "" "$E5"
