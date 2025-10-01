import time
import sys
import random

try:
    import cupy as cp
    # 使用するGPUデバイスの情報を取得
    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
    print(f"GPU検出: {device_name} (cupy)")
except ImportError:
    print("エラー: cupyがインストールされていません。")
    print("`pip install cupy-cudaXXX` (XXXはCUDAバージョン) を実行してください。")
    sys.exit(1)
except Exception as e:
    print(f"エラー: GPUの初期化に失敗しました。: {e}")
    sys.exit(1)


try:

    ARRAY_SIZE = int(input("配列サイズ (ARRAY_SIZE) を入力してください: "))
except (ValueError, KeyboardInterrupt):
    print("\n入力が中断されたか、無効な値です。終了します。")
    sys.exit(0)

# パフォーマンス測定用のバッチサイズの候補リスト
TEST_BATCHES = list(range(100_000, 1_000_000, 100_000)) + list(range(1_000_000, 10_000_001, 1_000_000))

# GPU用配列を準備
sorted_arr = cp.arange(ARRAY_SIZE, dtype=cp.int32).reshape(1, -1)

def try_batch(batch_size, sync=False):
    try:
        start = time.time()
        rnd = cp.random.rand(batch_size, ARRAY_SIZE)
        idx = cp.argsort(rnd, axis=1)
        _ = cp.take_along_axis(sorted_arr, idx, axis=1)
        if sync:
            cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        return batch_size / elapsed if elapsed > 0 else 0
    except cp.cuda.memory.OutOfMemoryError:
        cp._default_memory_pool.free_all_blocks()
        return None
    except Exception as e:
        print(f"バッチ試行中にエラーが発生しました: {e}")
        raise

def find_optimal_batch():
    """最適なバッチサイズを探索する"""
    print("最適なバッチサイズを探索中...")
    best_speed, best_batch = 0, 0
    for b in TEST_BATCHES:
        print(f"  試行中: {b:10,d} ...", end="", flush=True)
        speed = try_batch(b, sync=False)
        if speed is None:
            print(" メモリ不足")
            break
        print(f" 速度: {speed:,.0f} 回/秒")
        if speed > best_speed:
            best_speed, best_batch = speed, b
        else:
            print(f"\n探索完了。最適バッチサイズを採用: {best_batch:,} (ピーク速度: {best_speed:,.0f} 回/秒)\n")
            return best_batch
    if best_batch == 0:
        print("\nエラー: 実行可能なバッチサイズが見つかりませんでした。メモリが不足している可能性があります。")
        sys.exit(1)
    print(f"\n探索完了。最大バッチサイズを採用: {best_batch:,} (ピーク速度: {best_speed:,.0f} 回/秒)\n")
    return best_batch

def main():
    """メインの計算処理"""
    print("\n" + "=" * 50)
    print("ボゴソートを開始します。")
    target_list_str = str(sorted_arr.get().tolist()[0])
    print(f"目標の状態: {target_list_str}")
    initial_rnd = cp.random.rand(1, ARRAY_SIZE)
    initial_idx = cp.argsort(initial_rnd, axis=1)
    initial_shuffled = cp.take_along_axis(sorted_arr, initial_idx, axis=1)
    initial_list = initial_shuffled.get().tolist()[0]
    print(f"開始時の状態: {str(initial_list)}")
    print("=" * 50 + "\n")

    batch_size = find_optimal_batch()

    total_count = 0
    start_time = time.time()
    try:
        print("計算を開始します... (Ctrl+Cで中断)")
        while True:
            rnd = cp.random.rand(batch_size, ARRAY_SIZE)
            idx = cp.argsort(rnd, axis=1)
            shuffled = cp.take_along_axis(sorted_arr, idx, axis=1)

            results = cp.all(shuffled == sorted_arr, axis=1)
            found = cp.any(results)


            total_count += batch_size

            if found:
                # バッチ内で最初に成功した配列のインデックスをGPU上で取得
                found_index_gpu = cp.argmax(results)
                
                found_index_cpu = found_index_gpu.item()
                
                # 正確な試行回数を計算
                exact_count = (total_count - batch_size) + (found_index_cpu + 1)

                elapsed = time.time() - start_time
                speed = total_count / elapsed if elapsed > 0 else 0
                
                print("\n" + "-" * 50)
                print(f"成功！ {exact_count:,} 回目でソート済み配列を発見しました。")
                print(f"  - (総チェック回数: {total_count:,} 回)") 
                print(f"  - 経過時間: {elapsed:.4f} 秒")
                print(f"  - 平均速度: {speed:,.2f} 回/秒")
                print("-" * 50)
                break
            

            if total_count % (batch_size * 50) == 0:
                elapsed = time.time() - start_time
                speed = total_count / elapsed if elapsed > 0 else 0
                sample_idx = random.randint(0, batch_size - 1)
                sample_gpu = shuffled[sample_idx]
                sample_list = sample_gpu.get().tolist()
                if ARRAY_SIZE <= 20:
                    sample_str = str(sample_list)
                else:
                    head = ", ".join(map(str, sample_list[:5]))
                    tail = ", ".join(map(str, sample_list[-5:]))
                    sample_str = f"[{head}, ..., {tail}]"
                
                print(f"試行回数: {total_count:15,d} 回 ({speed:12,.0f} 回/秒) | サンプル: {sample_str}")

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        speed = total_count / elapsed if elapsed > 0 else 0
        print("\n" + "-" * 50)
        print("計算が中断されました。")
        print(f"  - 実行回数: {total_count:,} 回") 
        print(f"  - 経過時間: {elapsed:.4f} 秒")
        print(f"  - 平均速度: {speed:,.2f} 回/秒")
        print("-" * 50)

if __name__ == "__main__":
    main()