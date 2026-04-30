# RoboTwin 컨테이너 환경 트러블슈팅

컨테이너 재생성 시 동일한 문제가 발생할 수 있습니다. 아래 순서대로 조치하세요.

---

## 환경 정보

- OS: Ubuntu 24.04 (Noble), glibc 2.39
- NVIDIA 드라이버: 580.95.05 (호스트에서 bind-mount로 주입됨)
- GPU: NVIDIA B200 (Device Minor: 1, 즉 `/dev/nvidia1`, `/dev/nvidia0` 없음)
- Conda 환경: `RoboTwin` (Python 3.10)
- 주의: `/tmp`가 `tmpfs (noexec)`로 마운트되어 있어 `/tmp`에서 `.so` 실행 불가

---

## 문제 1: `ModuleNotFoundError: No module named 'pkg_resources'`

**원인**: `setuptools` 82.x에서 `pkg_resources`가 독립 모듈로 분리되어 제거됨.

**증상**:
```
import sapien.core as sapien
ModuleNotFoundError: No module named 'pkg_resources'
```

**조치**:
```bash
conda run -n RoboTwin pip install "setuptools<81" --force-reinstall
```

---

## 문제 2: `Render Error` (SAPIEN Vulkan 렌더러 초기화 실패)

**원인**: 두 가지 문제가 복합적으로 발생.

### 2-1. `collect_data.sh` 실행 시 base Python 사용

`collect_data.sh`는 conda 환경이 활성화된 상태에서 실행해야 합니다:
```bash
conda activate RoboTwin
bash collect_data.sh beat_block_hammer demo_clean 0
```

### 2-2. NVIDIA GL/Vulkan 라이브러리 누락

컨테이너에는 CUDA/compute 라이브러리만 bind-mount되고, GL/Vulkan 라이브러리(`libGLX_nvidia.so.0` 등)는 없습니다.

**진단**:
```
ERROR: libGLX_nvidia.so.0: cannot open shared object file: No such file or directory
Failed to find Vulkan ICD file.
```

**⚠️ 경고: `apt-get install libnvidia-gl-580`은 절대 사용하지 마세요.**
저장소의 최신 버전(예: 580.126.20)이 설치되면 호스트 드라이버 버전(580.95.05)과 불일치하여 apt 의존성이 깨집니다.
반드시 아래 방식으로 **drb 수동 추출 + 파일 직접 복사**만 사용하세요.

**조치 A**: 드라이버 버전과 일치하는 `libnvidia-gl` deb를 수동으로 추출하여 설치

```bash
# 1. 현재 호스트 드라이버 버전 확인
nvidia-smi | grep "Driver Version"
# → 예: 580.95.05

# 2. 이 버전에 해당하는 apt 패키지 버전 문자열 확인
apt-cache policy libnvidia-gl-580 | grep "580.95.05"
# → 예: 580.95.05-0ubuntu1  (이 문자열을 아래 명령에 그대로 사용)
# ※ 버전 문자열이 없으면 apt update 후 재시도

# 3. deb 다운로드 후 추출 (apt install 사용 금지 — 파일만 내려받음)
cd /tmp
apt-get download libnvidia-gl-580=580.95.05-0ubuntu1
mkdir -p /tmp/nvidia-gl-extract
dpkg-deb --extract /tmp/libnvidia-gl-580_580.95.05-0ubuntu1_amd64.deb /tmp/nvidia-gl-extract/

# bind-mount된 파일 목록 추출 (덮어쓰면 안 됨)
mount | grep "x86_64-linux-gnu" | awk '{print $3}' | xargs -I{} basename {} > /tmp/bindmount_files.txt

# bind-mount되지 않은 파일만 복사 (gcsudo 필요)
cd /tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu/
for f in *; do
  if ! grep -qxF "$f" /tmp/bindmount_files.txt; then
    gcsudo cp -P "$f" /usr/lib/x86_64-linux-gnu/
  fi
done
gcsudo cp -rP /tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu/nvidia /usr/lib/x86_64-linux-gnu/nvidia 2>/dev/null
gcsudo ldconfig
```

> **주의**: `apt-get install`로는 설치 불가 — `/tmp`(tmpfs)와 `/usr`(overlay)가 다른 파일시스템이라 dpkg의 hard link 백업 단계에서 `Invalid cross-device link` 오류 발생.

### 2-3. glibc 호환성 심볼 누락 (핵심 문제)

`libnvidia-glcore.so.580.95.05`가 glibc 2.34+에서 제거된 심볼들을 참조합니다:
- `__malloc_hook`, `__realloc_hook`, `__free_hook`, `__memalign_hook` (glibc 2.34에서 제거)
- `ErrorF`, `xf86ProcessOptions` (X.Org 서버 내부 심볼)

**진단**:
```bash
LD_DEBUG=libs python3 -c "import ctypes; ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0')" 2>&1 | grep "error:"
# → symbol lookup error: undefined symbol: __malloc_hook (fatal)
# → symbol lookup error: undefined symbol: ErrorF (fatal)
```

**조치 B**: 호환성 shim 라이브러리 빌드 및 설치

```bash
# 1. shim 소스 작성
cat > /tmp/malloc_hook_shim.c << 'EOF'
#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>

/* glibc 2.34+에서 제거된 malloc hook 호환 shim */
void *(*__malloc_hook)(size_t, const void *) __attribute__((visibility("default"))) = NULL;
void *(*__realloc_hook)(void *, size_t, const void *) __attribute__((visibility("default"))) = NULL;
void (*__free_hook)(void *, const void *) __attribute__((visibility("default"))) = NULL;
void *(*__memalign_hook)(size_t, size_t, const void *) __attribute__((visibility("default"))) = NULL;

/* X.Org 서버 내부 심볼 shim */
void ErrorF(const char *f, ...) __attribute__((visibility("default")));
void ErrorF(const char *f, ...) { (void)f; }

void xf86ProcessOptions(void) __attribute__((visibility("default")));
void xf86ProcessOptions(void) {}

void miCreateDefColormap(void) __attribute__((visibility("default")));
void miCreateDefColormap(void) {}
EOF

# 2. shim 빌드 (conda env lib에 저장 — noexec 아닌 경로)
gcc -shared -fPIC \
    -o /home/yonsei_meat/miniconda3/envs/RoboTwin/lib/libnvidia_malloc_compat.so \
    /tmp/malloc_hook_shim.c

# 3. 빌드 확인
echo $?  # 0이면 성공
```

> **주의**: `/tmp`는 `noexec`이므로 `.so`를 그곳에 빌드해도 `LD_PRELOAD`로 로드 불가. conda 환경 lib 경로에 빌드해야 함.

### 2-4. `collect_data.sh`에 LD_PRELOAD 영구 적용

[collect_data.sh](collect_data.sh)에 `LD_PRELOAD` 설정이 이미 추가되어 있습니다:

```bash
export LD_PRELOAD=/home/yonsei_meat/miniconda3/envs/RoboTwin/lib/libnvidia_malloc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}
```

컨테이너 재생성 후에는 **조치 B의 shim 빌드만 다시 실행**하면 됩니다 (소스 파일은 위 내용 그대로).

---

## 문제 3: `CUDA error: no kernel image is available for execution on the device` (CuroboPlanner import 실패)

**원인**: NVIDIA B200은 compute capability sm_100 (Blackwell 아키텍처)이지만, 기존에 설치된 PyTorch 2.4.1+cu121은 최대 sm_90까지만 지원함.

**증상**:
```
[planner.py]: Something wrong happened when importing CuroboPlanner!
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
에러는 curobo의 `normalize_quaternion` TorchScript 함수가 B200에서 CUDA 커널을 실행하려 할 때 발생.
결과적으로 `CuroboPlanner` 클래스 정의 자체가 실패하여 `ImportError: cannot import name 'CuroboPlanner'`로 이어짐.

**진단**:
```bash
conda run -n RoboTwin python -c "import torch; print(torch.cuda.get_device_capability())"
# → (10, 0)  ← sm_100, B200

conda run -n RoboTwin python -c "import torch; print(torch.__version__)"
# → 2.4.1+cu121  ← sm_50~sm_90만 지원
```

**조치**: PyTorch를 CUDA 12.8 빌드(cu128)로 업그레이드 후 curobo 재빌드

```bash
# 1. PyTorch 2.7.0+cu128 설치 (sm_100 Blackwell 지원)
conda run -n RoboTwin pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2. 설치 확인 — 경고 없이 CUDA 연산이 성공해야 함
conda run -n RoboTwin python -c \
    "import torch; x = torch.tensor([1.0]).cuda(); print('OK:', x)"
# → OK: tensor([1.], device='cuda:0')

# 3. curobo 소스 재빌드 (새 PyTorch 버전에 맞게)
cd /NHNHOME/WORKSPACE/0526040040_A/sunghyun/Soft-VLA/third_party/RoboTwin/envs/curobo
conda run -n RoboTwin pip install -e . --no-build-isolation

# 4. import 확인
cd /NHNHOME/WORKSPACE/0526040040_A/sunghyun/Soft-VLA/third_party/RoboTwin
conda run -n RoboTwin python -c "from envs.robot.planner import CuroboPlanner; print('OK')"
# → CuroboPlanner import OK
```

> **참고**: 시스템에 CUDA 12.8 (`nvcc --version` 확인)이 설치되어 있어야 cu128 빌드와 호환됨.
> cu128에서 요구하는 nvidia-cudnn, nvidia-cusolver 등 CUDA 서브 패키지는 pip가 자동으로 업그레이드함.

---

## 문제 4: OIDN 디노이저 비활성화로 학습 데이터 이미지 품질 저하

**원인**: NVIDIA B200(Blackwell, sm_100)에서 OIDN CUDA 백엔드가 미지원. OIDN 디바이스 초기화 실패로 렌더러가 디노이저 없이 동작.

**증상**:
```
[svulkan2] [error] OIDN Error: invalid handle
```
`collect_data.sh`로 실행 시(`CUDA_VISIBLE_DEVICES` 설정 상태)에만 발생. `python script/collect_data.py ...`로 직접 실행 시에는 다른 처리 경로를 타서 미발생.

**영향**:
- `set_ray_tracing_samples_per_pixel(32)`의 ray-traced 이미지는 spp가 낮아 노이즈가 많음
- OIDN이 꺼지면 노이즈가 제거되지 않은 채 렌더링됨
- **HDF5에 저장되는 학습 데이터 RGB 이미지에도 노이즈가 그대로 포함됨**
- optix 디노이저로 대체해도 동일하게 검정 화면 출력되어 사용 불가

**원인 분석**: SAPIEN의 `set_ray_tracing_denoiser("oidn")`은 `'none'`/`'oidn'`/`'optix'`만 지원하며 CPU 지정 불가.
OIDN 2.0.1은 CUDA 디바이스 플러그인(`libOpenImageDenoise_device_cuda.so.2.0.1`)을 우선 로드하는데,
B200(sm_100) 지원 커널이 없어 "invalid handle"로 초기화 실패. `OIDN_DEFAULT_DEVICE` 환경변수는 OIDN 2.0.1에서 미지원.

**조치**: OIDN CUDA 플러그인을 비활성화하여 CPU 폴백 강제

```bash
# CUDA 플러그인 비활성화 (.disabled로 rename — 삭제 금지)
mv /home/yonsei_meat/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library/libOpenImageDenoise_device_cuda.so.2.0.1 \
   /home/yonsei_meat/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library/libOpenImageDenoise_device_cuda.so.2.0.1.disabled
```

OIDN이 CUDA 플러그인을 찾지 못하면 코어 라이브러리의 CPU 백엔드로 자동 폴백됨.

> **참고**: OIDN CPU 백엔드는 속도가 느리지만 정확한 노이즈 제거를 보장함. 프레임당 수십~수백 ms 추가될 수 있으나 데이터 품질 유지를 위해 필요.
> **컨테이너 재생성 후**: 위 mv 명령을 다시 실행해야 함 (conda 환경 내 sapien 패키지가 재설치되면 초기화됨).

---

## 컨테이너 재생성 후 체크리스트

```bash
# 1. setuptools 다운그레이드
conda run -n RoboTwin pip install "setuptools<81" --force-reinstall

# 2. NVIDIA GL 라이브러리 설치 (드라이버 버전 확인 후 진행)
nvidia-smi | grep "Driver Version"
# → 버전에 맞게 위 "조치 A" 수행

# 3. 호환성 shim 빌드
# → 위 "조치 B" 수행

# 4. OIDN CUDA 플러그인 비활성화 (CPU 폴백 강제)
mv /home/yonsei_meat/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library/libOpenImageDenoise_device_cuda.so.2.0.1 \
   /home/yonsei_meat/miniconda3/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library/libOpenImageDenoise_device_cuda.so.2.0.1.disabled

# 5. PyTorch cu128 업그레이드 + curobo 재빌드
conda run -n RoboTwin pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
cd /NHNHOME/WORKSPACE/0526040040_A/sunghyun/Soft-VLA/third_party/RoboTwin/envs/curobo
conda run -n RoboTwin pip install -e . --no-build-isolation

# 6. 렌더러 및 CuroboPlanner 동작 확인
conda activate RoboTwin
python script/collect_data.py beat_block_hammer demo_clean 2>&1 | head -5
# → "Render Well" 또는 데이터 수집 시작이면 성공
```

---

## 참고: apt install이 실패하는 이유 및 apt가 망가지는 경우

### apt install이 실패하는 이유

이 컨테이너에서 `apt-get install libnvidia-gl-580`은 아래 이유로 실패합니다:

1. `/tmp`(tmpfs)와 `/usr`(overlay)이 다른 파일시스템 → dpkg hard link 백업 불가 (`Invalid cross-device link`)
2. 일부 NVIDIA 파일들이 호스트에서 read-only bind-mount → `rm`, `mv` 불가 (`Device or resource busy`)

따라서 `dpkg-deb --extract`로 수동 추출 후 파일만 복사하는 방식을 사용합니다.

### ⚠️ apt가 망가지는 경우 (반드시 피할 것)

`apt-get install libnvidia-gl-580`을 그냥 실행하면 저장소 최신 버전(예: 580.126.20)이 설치 시도됩니다.
`libnvidia-gpucomp-580`, `libnvidia-compute-580` 등 의존 패키지도 최신 버전으로 당겨오는데,
이들 일부는 `Invalid cross-device link`로 설치 실패하고 **절반만 설치된 상태**가 됩니다.

그러면 apt 의존성 DB에는 580.126.20 버전이 필요하다고 기록되지만 실제로는 설치되지 않아:
```
libnvidia-decode-580 : Depends: libnvidia-compute-580 (= 580.126.20-1ubuntu1) but it is not installed
E: Unmet dependencies. Try 'apt --fix-broken install'
```
이 상태가 됩니다.

**복구 방법** (이미 망가진 경우):
```bash
# 절반 설치된 패키지 강제 제거
gcsudo dpkg --remove --force-depends \
  libnvidia-gl-580 libnvidia-compute-580 libnvidia-decode-580 \
  libnvidia-gpucomp-580 libnvidia-cfg1-580 libnvidia-common-580 \
  libnvidia-egl-gbm1 libnvidia-egl-wayland1 libnvidia-egl-xcb1 \
  libnvidia-egl-xlib1 nvidia-persistenced

# 의존성 복구
gcsudo apt --fix-broken install -y
```

---

## 문제 5: `nvidia-smi` 실패 (`Failed to initialize NVML: Driver/library version mismatch`)

**증상**:
```bash
nvidia-smi
# Failed to initialize NVML: Driver/library version mismatch
# NVML library version: 590.48
```

커널 드라이버는 580인데, 일부 user-space 라이브러리 symlink가 590으로 향할 때 발생합니다.

**당시 확인된 상태**:
- 커널 드라이버: `580.95.05` (`/proc/driver/nvidia/version`)
- `libnvidia-ml.so.1` -> `libnvidia-ml.so.590.48.01` (오염)
- 아래 symlink들도 590으로 연결되어 있었음:
  - `libnvidia-nvvm.so.4`
  - `libnvidia-opencl.so.1`
  - `libnvidia-ptxjitcompiler.so.1`

### 조치: symlink를 580.95.05로 복구

```bash
gcsudo ln -sf libnvidia-ml.so.580.95.05 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
gcsudo ln -sf libnvidia-nvvm.so.580.95.05 /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4
gcsudo ln -sf libnvidia-opencl.so.580.95.05 /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
gcsudo ln -sf libnvidia-ptxjitcompiler.so.580.95.05 /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1

nvidia-smi
# NVIDIA-SMI 580.95.05 / Driver Version: 580.95.05 로 정상 출력
```

### 왜 이렇게 바뀌었는지 (추적 가능한 원인)

가능성이 높은 순서:
1. 컨테이너 내부에서 `apt install`/`apt upgrade`가 수행되면서 일부 NVIDIA user-space 라이브러리(590 계열)만 갱신됨.
2. 호스트에서 bind-mount된 커널 드라이버(580)와 컨테이너 라이브러리(590)가 섞여 mismatch 발생.
3. `apt --fix-broken install` 같은 복구 과정에서 symlink가 최신 파일(590)로 재지정됨.

> 참고: `LD_PRELOAD` shim, OIDN 플러그인 on/off, setuptools/PyTorch 교체 자체는 NVML 버전을 직접 바꾸지 않습니다.
> NVML mismatch의 직접 원인은 대체로 `/usr/lib/x86_64-linux-gnu/libnvidia-*.so*` 체인의 버전 불일치입니다.
