import time
import sys

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

# ---- 設定 ----
try:
    # ユーザーに配列サイズを入力させる
    ARRAY_SIZE = int(input("配列サイズ (ARRAY_SIZE) を入力してください: "))
except (ValueError, KeyboardInterrupt):
    print("\n入力が中断されたか、無効な値です。終了します。")
    sys.exit(0)

# パフォーマンス測定用のバッチサイズの候補リスト
TEST_BATCHES = list(range(100_000, 1_000_000, 100_000)) + list(range(1_000_000, 10_000_001, 1_000_000))

# ==== GPU用配列を準備（メインループで再利用する） ====
# 0から始まる連番の配列を作成し、(1, ARRAY_SIZE)の形状に変形
sorted_arr = cp.arange(ARRAY_SIZE, dtype=cp.int32).reshape(1, -1)

def try_batch(batch_size, sync=False):
    """指定されたバッチサイズでシャッフル処理の速度を測定する"""
    try:
        start = time.time()

        # ランダムな値を生成
        rnd = cp.random.rand(batch_size, ARRAY_SIZE)
        # ランダムな値に基づいてインデックスをソート
        idx = cp.argsort(rnd, axis=1)
        # ソートされたインデックスを使って元の配列をシャッフル
        _ = cp.take_along_axis(sorted_arr, idx, axis=1)

        if sync:
            # GPU処理の完了を待つ（正確な時間測定のため）
            cp.cuda.Stream.null.synchronize()

        elapsed = time.time() - start
        # 経過時間が0より大きい場合、秒間処理回数を計算
        return batch_size / elapsed if elapsed > 0 else 0
    except cp.cuda.memory.OutOfMemoryError:
        # GPUメモリが不足した場合、メモリプールを解放してNoneを返す
        cp._default_memory_pool.free_all_blocks()
        return None
    except Exception as e:
        # その他の予期せぬエラー
        print(f"バッチ試行中にエラーが発生しました: {e}")
        raise

def find_optimal_batch():
    """最適なバッチサイズを探索する"""
    print("最適なバッチサイズを探索中...")
    best_speed, best_batch = 0, 0

    for b in TEST_BATCHES:
        print(f"  試行中: {b:10,d} ...", end="", flush=True)
        # sync=Falseで測定し、おおよその速度を計測（オーバーヘッド削減）
        speed = try_batch(b, sync=False)

        if speed is None:
            # メモリ不足で実行できなかった場合
            print(" メモリ不足")
            break

        print(f" 速度: {speed:,.0f} 回/秒")
        if speed > best_speed:
            # より速いバッチサイズが見つかった場合
            best_speed, best_batch = speed, b
        else:
            # 速度が低下し始めたら、その前のバッチサイズが最適と判断
            print(f"\n探索完了。最適バッチサイズを採用: {best_batch:,} (ピーク速度: {best_speed:,.0f} 回/秒)\n")
            return best_batch

    if best_batch == 0:
        print("\nエラー: 実行可能なバッチサイズが見つかりませんでした。メモリが不足している可能性があります。")
        sys.exit(1)
        
    print(f"\n探索完了。最大バッチサイズを採用: {best_batch:,} (ピーク速度: {best_speed:,.0f} 回/秒)\n")
    return best_batch

def main():
    """メインの計算処理"""
    batch_size = find_optimal_batch()

    total_count = 0
    start_time = time.time()
    try:
        print("計算を開始します... (Ctrl+Cで中断)")
        while True:
            # ランダムな値を生成
            rnd = cp.random.rand(batch_size, ARRAY_SIZE)
            # ランダムな値に基づいてインデックスをソート
            idx = cp.argsort(rnd, axis=1)
            # ソートされたインデックスを使って元の配列をシャッフル
            shuffled = cp.take_along_axis(sorted_arr, idx, axis=1)
            
            # シャッフル後の配列が元のソート済み配列と一致するかチェック
            # cp.allは各行が一致するかを判定し、cp.anyはいずれかの行が一致するかを判定
            found = cp.any(cp.all(shuffled == sorted_arr, axis=1))
            
            total_count += batch_size

            if found:
                # 一致する配列が見つかった場合
                elapsed = time.time() - start_time
                speed = total_count / elapsed if elapsed > 0 else 0
                print("-" * 50)
                print(f"成功！ {total_count:,} 回目でソート済み配列を発見しました。")
                print(f"  - 経過時間: {elapsed:.4f} 秒")
                print(f"  - 平均速度: {speed:,.2f} 回/秒")
                print("-" * 50)
                break
            
            # 定期的に進捗を表示
            if total_count % (batch_size * 50) == 0:
                elapsed = time.time() - start_time
                speed = total_count / elapsed if elapsed > 0 else 0
                print(f"進捗: {total_count:,} 回 ({speed:,.0f} 回/秒)")

    except KeyboardInterrupt:
        # ユーザーによる中断 (Ctrl+C)
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
