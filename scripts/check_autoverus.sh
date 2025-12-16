#!/usr/bin/env bash
# which version of verus works to verify autoverus?!
# run this script to find out if a given install works

set -euo pipefail

ROOT="repositories/verus-proof-synthesis/benchmarks"

# Collect all *.rs files under directories containing "verified"
FILES=$(find "$ROOT" -type d -name verified -print0 | \
        xargs -0 -I{} find "{}" -maxdepth 1 -type f -name "*.rs")

total=0
passed=0
failed=0

echo "=== Running Verus on verified AutoVerus benchmark files ==="
echo

for file in $FILES; do
    total=$((total + 1))
    echo "[checking] $file"

    if verus "$file" >/dev/null 2>&1; then
        echo "    ✔ PASS"
        passed=$((passed + 1))
    else
        echo "    ✘ FAIL"
        failed=$((failed + 1))
    fi
done

echo
echo "=== Summary ==="
echo "Total files:  $total"
echo "Passed:       $passed"
echo "Failed:       $failed"

# exit with nonzero status if any failed
if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
