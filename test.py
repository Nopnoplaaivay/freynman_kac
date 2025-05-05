import numpy as np
import pandas as pd
import time
from src.methods import (
    monte_carlo_price,
    finite_difference_price,
    method_of_lines_price,
    spectral_method_price,
    fem_price,
    deep_ritz_price,
    deeponet_price,
    fourier_neural_operator_price,
    quantum_price_simulation
)
import concurrent.futures
import os
from scipy.stats import norm


METHODS = {
    "Monte Carlo": monte_carlo_price,
    "Finite Difference": finite_difference_price,
    "Method of Lines": method_of_lines_price,
    "Spectral Method": spectral_method_price,
    "Finite Elements": fem_price,
    "Deep Ritz": deep_ritz_price,
    "DeepONet": deeponet_price,
    "Fourier Neural Operator": fourier_neural_operator_price,
    "Quantum Simulation": quantum_price_simulation
}

def black_scholes_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# Thay đổi hàm run để chỉ chạy 1 lần (không có vòng lặp num_runs)
def run_single(name, func, true_price, S0, K, r, sigma, T):
    # kwargs definitions có thể đặt ở đây hoặc truyền vào
    fd_kwargs = {"S_max": 300, "M": 200, "N": 200}
    mol_kwargs = {"S_max": 300, "M": 200}
    spec_kwargs = {"S_max": 300, "N": 50}
    fem_kwargs = {"S_max": 300, "M": 200, "N": 200}
    deep_ritz_kwargs = {"S_max": 300, "num_collocation": 1000, "num_epochs": 5000, "lr": 1e-3}
    deeponet_kwargs = {"S_max": 300, "num_samples": 1000, "epochs": 1000, "lr": 1e-3, "m": 100, "p": 50}
    fourier_neural_operator_kwargs = {"S_max": 300, "num_samples": 1000, "epochs": 1000, "lr": 1e-3, "m": 100, "modes": 16, "width": 64, "layers": 4}

    start = time.time()
    price = 0 # Giá trị mặc định

    # Logic gọi func (tương tự hàm run cũ nhưng không có vòng lặp)
    try: # Thêm try-except để bắt lỗi nếu có
        if name == "Monte Carlo":
            price = func(S0, K, r, sigma, T, N=1000000)
        elif name == "Method of Lines":
            price = func(S0, K, r, sigma, T, **mol_kwargs)
        elif name == "Finite Difference":
            price = func(S0, K, r, sigma, T, **fd_kwargs)
        elif name == "Spectral Method":
            price = func(S0, K, r, sigma, T, **spec_kwargs)
        elif name == "Finite Elements":
            price = func(S0, K, r, sigma, T, **fem_kwargs)
        elif name == "Deep Ritz":
            price = func(S0, K, r, sigma, T, **deep_ritz_kwargs)
        elif name == "DeepONet":
            price = func(S0, K, r, sigma, T, **deeponet_kwargs)
        elif name == "Fourier Neural Operator":
            price = func(S0, K, r, sigma, T, **fourier_neural_operator_kwargs)
        elif name == "Quantum Simulation":
            # Simulate the quantum circuit with a normalized state vector.
            state = np.array([0.7071, 0.7071, 0.0, 0.7071])
            price = func(state, r, T, shots=100)

        end = time.time()
        runtime = end - start
        error = abs(price - true_price)

        print(f"Finished: {name} S0={S0:.2f}, K={K:.2f} - Runtime: {runtime:.4f}s") # In ra để theo dõi
        # Trả về kết quả cho một lần chạy
        return {
            "S0": S0, "K": K, "r": r, "sigma": sigma, "T": T,
            "True Price": true_price, "Method": name,
            "Mean Price": price, "Std Price": 0.0, # Vì chỉ chạy 1 lần
            "Mean Error": error, "Std Error": 0.0,
            "Mean Runtime": runtime, "Std Runtime": 0.0
        }
    except Exception as e:
        print(f"ERROR running {name} for S0={S0:.2f}, K={K:.2f}: {e}")
        # Trả về giá trị lỗi hoặc None để biết tác vụ thất bại
        return {
            "S0": S0, "K": K, "r": r, "sigma": sigma, "T": T,
            "True Price": true_price, "Method": name,
            "Mean Price": np.nan, "Std Price": np.nan,
            "Mean Error": np.nan, "Std Error": np.nan,
            "Mean Runtime": np.nan, "Std Runtime": np.nan,
            "Error Info": str(e)
        }


# Hàm worker xử lý một bộ tham số
def process_parameter_set(params):
    S0, K, r, sigma, T, true_price = params
    results_for_set = []
    print(f"Processing set: S0={S0:.2f}, K={K:.2f}, r={r:.2f}, sigma={sigma:.2f}, T={T:.2f}")
    for name, func in METHODS.items():
        # Gọi hàm run_single đã được sửa đổi
        result_dict = run_single(name, func, true_price, S0, K, r, sigma, T)
        if result_dict: # Kiểm tra nếu không phải None (trường hợp lỗi)
             results_for_set.append(result_dict)
    return results_for_set

def main():
    n_sets = 50
    np.random.seed(42)

    # Generate parameter sets
    S0_values = np.random.uniform(80, 120, n_sets)
    K_values = np.random.uniform(80, 120, n_sets)
    r_values = np.random.uniform(0.01, 0.1, n_sets)
    sigma_values = np.random.uniform(0.1, 0.4, n_sets)
    T_values = np.random.uniform(0.25, 2.0, n_sets) # Tạm thời không dùng

    parameter_sets = []
    for i in range(n_sets):
        S0 = S0_values[i]
        K = K_values[i]
        r = r_values[i]
        sigma = sigma_values[i]
        T = 1.0 # Luôn là 1.0
        true_price = black_scholes_price(S0, K, r, sigma, T)
        parameter_sets.append((S0, K, r, sigma, T, true_price))

    results_list = []
    # Xác định số lượng workers (tiến trình) - thường bằng số core CPU
    # Có thể giảm nếu gặp vấn đề về bộ nhớ hoặc GPU contention
    num_workers = 4
    print(f"Using {num_workers} workers...")

    start_time_main = time.time()

    # Sử dụng ProcessPoolExecutor để chạy song song
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map sẽ gửi từng 'params' trong 'parameter_sets' đến hàm 'process_parameter_set'
        # và thu thập kết quả theo thứ tự
        future_results = executor.map(process_parameter_set, parameter_sets)

        # Thu thập kết quả từ các future
        for result_set in future_results:
            results_list.extend(result_set) # Nối list kết quả của mỗi bộ tham số vào list tổng

    end_time_main = time.time()
    print(f"\nTotal execution time: {end_time_main - start_time_main:.2f} seconds")

    # Convert results to a DataFrame
    df = pd.DataFrame(results_list)

    # Export to CSV
    df.to_csv("benchmark_results_parallel.csv", index=False)
    print("Results saved to benchmark_results_parallel.csv")


if __name__ == "__main__":
    # Đảm bảo code chạy đúng cách khi dùng multiprocessing trên Windows
    main()