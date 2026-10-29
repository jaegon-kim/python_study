#!/usr/bin/env bash
# =============================================================================
#
# 사용법:
#   터미널을 새로 열고:
#     bash start_llama_server.sh
#   서버가 백그라운드 docker 컨테이너로 기동되고,
#   본 스크립트가 자체적으로 health check 까지 마친 뒤 종료.
#
# 모델: Gemma 4 E2B-it Q4_K_M + mmproj-F16  (unsloth GGUF)
#       - 로컬 파일이 없으면 -hf 로 Hugging Face 에서 자동 다운로드
#       - thinking 모드는 비활성화 (에이전트/툴콜 워크로드 최적)
#       - KV cache 는 기본 bf16 유지 (Q4_K_M 1.7GB + ctx 4096 → T4 16GB 여유)
# 포트: 8181  (파이썬 SERVER_PORT 와 동일)
# =============================================================================

set -euo pipefail

CONTAINER_NAME="llama-server-project"
HOST_PORT=8181
MODEL_DIR="${HOME}/models"
HF_CACHE_DIR="${HOME}/.cache/huggingface"   # llama-server -hf uses huggingface_hub
                                            # under the hood, NOT LLAMA_CACHE.
                                            # 컨테이너의 /root/.cache/huggingface 에 마운트.
IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"

GGUF_FILENAME="gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_FILENAME="mmproj-BF16.gguf"           # unsloth 레포 현재 BF16 (F16 아님)
HF_REPO="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"

# -----------------------------------------------------------------------------
# 1. 사전 점검: docker 실행 가능 여부
# -----------------------------------------------------------------------------
echo "==> Docker 및 GPU 점검"
if ! command -v docker &>/dev/null; then
    echo "  ✗ docker 명령을 찾을 수 없습니다. 설치 상태 확인 필요." >&2
    exit 1
fi
docker info &>/dev/null || { echo "  ✗ docker 데몬이 실행 중이 아닙니다." >&2; exit 1; }
echo "  ✓ docker OK"

mkdir -p "${MODEL_DIR}" "${HF_CACHE_DIR}"

# -----------------------------------------------------------------------------
# 2. 모델 파일 위치 확인 → 분기 처리 (파이썬 스크립트와 동일 로직)
# -----------------------------------------------------------------------------
echo "==> 모델 파일 탐색 (${MODEL_DIR})"
LOCAL_MODEL="${MODEL_DIR}/${GGUF_FILENAME}"
LOCAL_MMPROJ="${MODEL_DIR}/${MMPROJ_FILENAME}"

MODEL_ARGS=()
if [[ -f "${LOCAL_MODEL}" && -f "${LOCAL_MMPROJ}" ]]; then
    echo "  📁 로컬 모델 파일 감지됨. 오프라인 모드로 실행"
    echo "     - 모델:   ${LOCAL_MODEL}"
    echo "     - mmproj: ${LOCAL_MMPROJ}"
    # 컨테이너 내부 경로로 변환 (MODEL_DIR 전체를 /models 로 마운트)
    MODEL_ARGS=(
        -m       "/models/${GGUF_FILENAME}"
        --mmproj "/models/${MMPROJ_FILENAME}"
    )
else
    echo "  🌐 로컬 모델이 없음. Hugging Face 자동 다운로드 모드(-hf)로 실행"
    echo "     - repo: ${HF_REPO}"
    echo "     - cache: ${HF_CACHE_DIR}  (HF 표준 캐시, 재실행 시 재사용됨)"
    MODEL_ARGS=( -hf "${HF_REPO}" )
fi

# -----------------------------------------------------------------------------
# 3. 기존 컨테이너 정리 (재실행 대비)
# -----------------------------------------------------------------------------
echo "==> 기존 ${CONTAINER_NAME} 컨테이너 정리"
docker rm -f "${CONTAINER_NAME}" &>/dev/null || true
echo "  ✓ 정리 완료"

# -----------------------------------------------------------------------------
# 4. 포트 충돌 확인
# -----------------------------------------------------------------------------
if ss -tln 2>/dev/null | grep -q ":${HOST_PORT} "; then
    echo "  ⚠ 포트 ${HOST_PORT} 이 이미 사용 중입니다." >&2
    echo "    다른 llama-server 인스턴스가 떠 있는지 확인:" >&2
    echo "      docker ps" >&2
    echo "      ss -tlnp | grep ${HOST_PORT}" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 5. 서버 기동 (detached)
#    인자는 파이썬 스크립트의 base_args + model_args 와 1:1 매칭
#    (컨테이너 내부 포트는 8080 고정, 호스트 측만 ${HOST_PORT} 로 매핑)
#
#    Gemma 4 관련:
#    - thinking 모드: chat template 에 의해 기본 ON. 에이전트/툴콜 수업에서는
#      OFF 가 안정적이라 --chat-template-kwargs '{"enable_thinking":false}' 추가.
#    - Google 권장 샘플링: temp=1.0, top-p=0.95, top-k=64.
#      본 스크립트는 수업 재현성을 위해 파이썬 스크립트와 동일한 --temp 0.1 유지.
#      에이전트 동작이 너무 결정론적이라 느껴지면 위 권장값으로 바꿔도 됨.
#    - CUDA 13.2 런타임은 GGUF 품질 저하 이슈가 알려져 있음 (unsloth 공지).
#      ghcr.io/ggml-org/llama.cpp:server-cuda 의 rolling tag 가 13.2 로 올라가면
#      문제가 생길 수 있음 → 학기 중 안정성 필요하면 digest 핀 권장:
#        IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:<digest>"
# -----------------------------------------------------------------------------
echo "==> llama-server 기동 (host port ${HOST_PORT} → container 8080)"
docker run -d --rm --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${HOST_PORT}:8080" \
    -v "${MODEL_DIR}:/models" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    "${IMAGE}" \
    "${MODEL_ARGS[@]}" \
    --host 0.0.0.0 \
    --port 8080 \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' \
    -fa auto \
    -ngl 99 \
    -c 4096 \
    -b 256 \
    --parallel 1 \
    --alias gemma \
    --temp 0.1 \
    > /dev/null

echo "  ✓ 컨테이너 시작됨"
echo

# -----------------------------------------------------------------------------
# 6. 상태 표시
# -----------------------------------------------------------------------------
sleep 2
docker ps --filter "name=${CONTAINER_NAME}" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo

# -----------------------------------------------------------------------------
# 7. 헬스 체크 (다운로드 상황 고려하여 최대 300초)
# -----------------------------------------------------------------------------
echo "==> 서버 헬스 체크 (최대 300초 대기, 모델 로드/다운로드 시간 포함)"
for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${HOST_PORT}/health" &>/dev/null; then
        echo "  ✅ llama-server 실행 완료! (http://127.0.0.1:${HOST_PORT})"
        # 채팅 템플릿 확인 (파이썬 스크립트 마지막 부분과 동일)
        TPL=$(curl -s "http://127.0.0.1:${HOST_PORT}/props" \
              | python3 -c "import sys,json;print(json.load(sys.stdin).get('chat_template','N/A')[:80])" 2>/dev/null || echo "N/A")
        echo "  📋 Chat template: ${TPL}..."
        break
    fi
    sleep 1
    if [[ $i -eq 300 ]]; then
        echo "  ❌ 서버 시작 실패. 로그 확인:" >&2
        echo "     docker logs ${CONTAINER_NAME}" >&2
        exit 1
    fi
done

echo

# =============================================================================
# 참고: Docker Model Runner (DMR) 대안
# -----------------------------------------------------------------------------
# unsloth 공식 문서는 더 간단한 1-line 실행으로 다음을 권장 (DMR 사용):
#
#   docker model run hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
#
# 차이점:
#   - DMR 은 Docker Desktop 의 model runner 기능 (별도 설치/활성화 필요)
#   - --jinja, --parallel, -b, --alias, --chat-template-kwargs 등
#     llama-server 의 세밀한 옵션은 DMR CLI 로는 노출되지 않음
#   - 빠르게 단일 모델만 띄워보고 싶을 때 참고용으로만 안내
# =============================================================================
