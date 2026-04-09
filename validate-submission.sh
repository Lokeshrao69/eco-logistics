#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Eco-Logistics — Pre-Submission Validation Script
# Run: bash validate-submission.sh
# ═══════════════════════════════════════════════════════════════

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass_count=0
fail_count=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; ((pass_count++)); }
fail() { echo -e "  ${RED}✗${NC} $1"; ((fail_count++)); }

echo -e "${BOLD}════════════════════════════════════════${NC}"
echo -e "${BOLD}  Eco-Logistics Pre-Submission Validator${NC}"
echo -e "${BOLD}════════════════════════════════════════${NC}"
echo ""

# ── Step 1: Required Files ───────────────────────────────────
echo -e "${BOLD}Step 1/5: Checking required files${NC}"

REQUIRED_FILES="models.py env.py main.py baseline.py inference.py openenv.yaml pyproject.toml Dockerfile README.md server/app.py uv.lock"
for f in $REQUIRED_FILES; do
    if [ -f "$f" ]; then
        pass "$f exists"
    else
        fail "$f MISSING"
    fi
done

# Check files are not empty
for f in pyproject.toml server/app.py inference.py; do
    if [ -s "$f" ]; then
        pass "$f has content ($(wc -l < "$f") lines)"
    else
        fail "$f is EMPTY"
    fi
done

echo ""

# ── Step 2: Python Tests ────────────────────────────────────
echo -e "${BOLD}Step 2/5: Running environment tests${NC}"

if python3 test_env.py > /tmp/test_output.txt 2>&1; then
    pass "All environment tests passed"
else
    fail "Some environment tests failed"
    tail -5 /tmp/test_output.txt
fi

echo ""

# ── Step 3: Inference.py Checks ──────────────────────────────
echo -e "${BOLD}Step 3/5: Checking inference.py compliance${NC}"

if grep -q "from openai import OpenAI" inference.py; then
    pass "Uses OpenAI client"
else
    fail "Missing OpenAI client import"
fi

if grep -q "API_BASE_URL" inference.py && grep -q "MODEL_NAME" inference.py && grep -q "HF_TOKEN" inference.py; then
    pass "Has required env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN)"
else
    fail "Missing required environment variables"
fi

if grep -q "\[START\]" inference.py && grep -q "\[STEP\]" inference.py && grep -q "\[END\]" inference.py; then
    pass "Has structured logging ([START], [STEP], [END])"
else
    fail "Missing structured logging format"
fi

if grep -q "litellm" inference.py; then
    fail "Contains litellm (must use OpenAI client only)"
else
    pass "No litellm dependency"
fi

echo ""

# ── Step 4: Docker Build ────────────────────────────────────
echo -e "${BOLD}Step 4/5: Docker build${NC}"

if command -v docker &>/dev/null; then
    if docker build -t eco-logistics-test . -q > /dev/null 2>&1; then
        pass "Docker build succeeded"
    else
        fail "Docker build failed"
    fi
else
    echo "  - Docker not installed, skipping"
fi

echo ""

# ── Step 5: OpenEnv Validate ────────────────────────────────
echo -e "${BOLD}Step 5/5: OpenEnv validate${NC}"

if command -v openenv &>/dev/null; then
    if openenv validate > /dev/null 2>&1; then
        pass "openenv validate passed"
    else
        fail "openenv validate failed"
        openenv validate 2>&1 | tail -5
    fi
else
    echo "  - openenv CLI not installed, skipping (pip install openenv-core)"
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════${NC}"
total=$((pass_count + fail_count))
if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  All checks passed! ($pass_count/$total)${NC}"
    echo -e "${GREEN}${BOLD}  Your submission is ready.${NC}"
else
    echo -e "${RED}${BOLD}  $fail_count check(s) failed ($pass_count/$total passed)${NC}"
    echo -e "${RED}${BOLD}  Fix the issues above before submitting.${NC}"
fi
echo -e "${BOLD}════════════════════════════════════════${NC}"
echo ""

exit $fail_count