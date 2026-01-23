import time
import torch
import subprocess
import threading

CHECK_INTERVAL = 60        # 扫描间隔
ALLOCATE_GB = 30           # 每张 GPU 占用显存
MAT_DIM = 8192             # 计算规模（越大计算越重）
SLEEP_BETWEEN_COMPUTE = 0  # 计算循环 sleep，0 = 满负载

occupied_gpus = set()
lock = threading.Lock()


def get_gpus_without_process():
    gpu_count = torch.cuda.device_count()
    gpu_has_proc = {i: False for i in range(gpu_count)}

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_index,pid",
                "--format=csv,noheader"
            ],
            encoding="utf-8"
        )

        for line in result.strip().split("\n"):
            if line.strip():
                gpu_idx, _ = line.split(",")
                gpu_has_proc[int(gpu_idx.strip())] = True

    except subprocess.CalledProcessError:
        pass

    return [i for i, used in gpu_has_proc.items() if not used]


def occupy_gpu(gpu_idx):
    try:
        torch.cuda.set_device(gpu_idx)

        # ---------- 显存占用 ----------
        num_elements = ALLOCATE_GB * 1024**3 // 4
        mem_block = torch.empty(
            num_elements,
            device=f"cuda:{gpu_idx}",
            dtype=torch.float32
        )

        # ---------- 计算张量 ----------
        a = torch.randn(
            (MAT_DIM, MAT_DIM),
            device=f"cuda:{gpu_idx}",
            dtype=torch.float16
        )
        b = torch.randn_like(a)

        torch.cuda.synchronize()
        print(f"[OK] GPU {gpu_idx} 已占用 {ALLOCATE_GB}GB 显存并开始计算")

        # ---------- 持续计算 ----------
        while True:
            c = torch.matmul(a, b)

            # 防止计算被编译器/缓存优化掉
            a = c * 0.999 + a * 0.001

            if SLEEP_BETWEEN_COMPUTE > 0:
                time.sleep(SLEEP_BETWEEN_COMPUTE)

    except Exception as e:
        print(f"[FAIL] GPU {gpu_idx} 占用失败: {e}")


def main_loop():
    print("开始 GPU 监控（无进程即显存 + 计算占用）...")

    while True:
        idle_gpus = get_gpus_without_process()

        for gpu_idx in idle_gpus:
            with lock:
                if gpu_idx in occupied_gpus:
                    continue
                occupied_gpus.add(gpu_idx)

            print(f"[INFO] 发现 GPU {gpu_idx} 空闲，启动占用线程")
            t = threading.Thread(
                target=occupy_gpu,
                args=(gpu_idx,),
                daemon=True
            )
            t.start()

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main_loop()





